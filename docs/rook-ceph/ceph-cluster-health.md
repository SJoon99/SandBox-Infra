# Ceph 상태 점검 보고서

작성일: 2026-03-11
대상: `rook-ceph` 클러스터

## 1. 요약

현재 Ceph 클러스터는 `HEALTH_WARN` 상태이며, 단순 경고 수준이 아니라 실제 데이터 중복도가 크게 저하된 상태입니다.

핵심 문제는 용량 부족이 아닙니다. 전체 사용량은 낮지만, SSD 기반 풀들이 요구하는 복제 배치 조건을 현재 클러스터 토폴로지가 만족하지 못하고 있어 많은 PG가 장기간 `undersized`, `degraded` 상태로 남아 있습니다.

가장 중요한 원인은 SSD OSD가 사실상 `sandbox-4` 한 호스트에만 정상적으로 남아 있고, `tempnode-bf3`의 SSD OSD 2개가 내려가 있다는 점입니다.

## 2. 주요 증상

### 2.1 Ceph health 저하

`ceph status` 기준:

- `HEALTH_WARN`
- `364388/729921 objects degraded (49.922%)`
- `70 pgs degraded`
- `83 pgs undersized`
- `82 pgs not deep-scrubbed in time`
- `82 pgs not scrubbed in time`

즉, 전체 오브젝트의 약 절반이 의도한 복제 수를 만족하지 못하고 있습니다.

### 2.2 OSD 일부 미동작

`ceph status` 기준:

- 전체 OSD: `36`
- 동작 중 OSD: `34 up`
- 클러스터 포함 OSD: `34 in`

현재 OSD 2개가 정상 서비스에 참여하지 못하고 있습니다.

### 2.3 Rook CephCluster 비정상 수렴

`kubectl get cephcluster -n rook-ceph -o wide` 기준:

- `PHASE: Progressing`
- `HEALTH: HEALTH_WARN`
- `MESSAGE: failed to start ceph osds ... aborting OSD provisioning after waiting more than 20 minutes`

`kubectl describe cephcluster rook-ceph -n rook-ceph` 기준:

- `State: Error`
- OSD provisioning 관련 `ReconcileFailed` 이벤트가 반복 발생

즉, Rook 오퍼레이터도 현재 스토리지 상태를 정상적으로 수렴시키지 못하고 있습니다.

### 2.4 일부 Ceph 데몬 probe 이상

최근 이벤트 기준:

- `rook-ceph-mgr-b` startup probe 오류
- `rook-ceph-mds-ceph-filesystem-hot-a` liveness probe 오류
- RGW readiness probe timeout 발생

이 항목은 현재 중복도 저하의 직접 원인으로 보이지는 않지만, 운영 안정성이 좋지 않다는 신호입니다.

## 3. 확인한 근거

### 3.1 Ceph 서비스 자체는 살아 있음

`ceph status` 기준:

- MON: 3개 quorum 정상
- MGR: active 1, standby 1
- MDS: active 1, standby 1
- RGW: active 1

즉, 클러스터가 완전히 죽은 상태는 아니고 "살아 있으나 건강하지 않은 상태"입니다.

### 3.2 OSD 토폴로지

`ceph osd tree` 기준:

- `sandbox-1`: HDD OSD만 존재
- `sandbox-2`: HDD OSD만 존재
- `sandbox-4`: 다수의 SSD OSD 정상 동작
- `tempnode-bf3`: SSD OSD `osd.2`, `osd.4` 가 `down`

핵심:

- `osd.2` on `tempnode-bf3`: `down`
- `osd.4` on `tempnode-bf3`: `down`

즉, SSD 계층에서 정상 호스트가 사실상 `sandbox-4` 하나만 남아 있습니다.

### 3.3 Pool 설정

`ceph osd pool ls detail` 기준으로 다음 SSD 관련 풀들이 복제수 `size 2`를 사용 중입니다.

- `ceph-objectstore.rgw.control`
- `ceph-objectstore.rgw.meta`
- `ceph-objectstore.rgw.log`
- `ceph-objectstore.rgw.buckets.index`
- `ceph-objectstore.rgw.buckets.non-ec`
- `hot-pool-ssd`
- `ceph-filesystem-hot-metadata`
- `ceph-filesystem-hot-hot-data`
- `rgw-hot-pool-ssd`

### 3.4 CRUSH rule

`ceph osd crush rule dump` 기준으로 SSD 관련 풀들은 공통적으로 아래 구조를 사용합니다.

- `take default~ssd`
- `chooseleaf ... type host`

즉, 복제본을 단순히 다른 OSD에 두는 것이 아니라 "서로 다른 host"에 두도록 설계되어 있습니다.

### 3.5 PG 상세 상태

`ceph health detail` 및 `ceph pg dump_stuck undersized` 기준:

- 다수의 PG가 약 9주 동안 `stuck undersized`
- 상태는 주로 `active+undersized+degraded`
- acting set이 `[8]`, `[5]`, `[26]` 처럼 단일 OSD만 잡히는 케이스 다수 확인

이것은 Ceph가 두 번째 복제본을 배치할 다른 SSD host를 찾지 못하고 있다는 의미입니다.

## 4. 원인 분석

### 4.1 주원인

현재 SSD 기반 풀은 host 단위로 최소 2개의 복제 배치를 요구하지만, 실제 정상 SSD host는 1개뿐입니다.

구체적으로:

- SSD 풀의 replica 설정은 `size 2`
- CRUSH rule은 host 분산을 강제
- 현재 정상 SSD OSD는 대부분 `sandbox-4`에 집중
- `tempnode-bf3`의 SSD OSD 2개는 down 상태

결과:

- Ceph는 SSD 풀의 복제 수를 만족하지 못함
- PG가 `undersized` 상태로 장기 고착
- 데이터 중복도 저하
- scrub, deep-scrub도 정상적으로 돌지 못함

### 4.2 부가 원인 또는 운영 리스크

#### 4.2.1 OSD provisioning 반복 실패

Rook는 현재도 OSD 준비/반영 작업을 반복 시도하다 실패하고 있습니다.

메시지:

- `aborting OSD provisioning after waiting more than 20 minutes`

즉, 단순히 과거 장애 흔적이 아니라 현재도 수렴에 실패하고 있습니다.

#### 4.2.2 Object Store 설정 충돌 이력

최근 이벤트에서 아래 오류가 확인됐습니다.

- `object store dataPool and sharedPools.dataPool=rgw-hot-pool-ssd are mutually exclusive`

이 오류는 이후 reconcile 성공 이벤트가 있었으므로 현재 지속 장애의 직접 원인은 아닐 수 있습니다. 다만 최근 오브젝트스토어 설정 변경 중 충돌이 있었던 것으로 보이며, 별도 검토가 필요합니다.

#### 4.2.3 컨테이너 런타임 exec 관련 probe 오류

`mgr`, `mds` probe 오류 메시지에 아래 내용이 보였습니다.

- `possible container breakout detected`

이건 Ceph 데이터 배치 문제와는 별개로, 노드 런타임 또는 exec 환경 이슈 가능성을 의미합니다.

## 5. 영향 범위

### 5.1 데이터 안정성

- 현재 데이터가 전부 접근 불가능한 상태는 아님
- 하지만 의도된 복제 수준이 유지되지 않고 있음
- 추가 장애가 발생하면 실제 서비스 장애나 데이터 손실 위험이 커짐

### 5.2 특히 영향 가능성이 높은 영역

다음 SSD 기반 리소스를 사용하는 영역이 우선 위험합니다.

- RGW / Object Storage 관련 메타데이터 및 hot 경로
- CephFS hot metadata / hot data 풀
- `hot-pool-ssd` 기반 PVC 또는 워크로드

### 5.3 운영상 주의점

현재 상태에서는 `sandbox-4` 의존도가 매우 높습니다.

즉시 피해야 할 작업:

- `sandbox-4` 재부팅
- `sandbox-4` drain
- `sandbox-4` 디스크/OSD 관련 파괴적 변경

## 6. 해결 방안

### 6.1 최우선 조치

가장 먼저 `tempnode-bf3`의 SSD OSD 2개가 왜 내려갔는지 확인해야 합니다.

확인 대상:

- 노드 자체 상태
- 디스크 인식 여부
- OSD prepare job 로그
- OSD pod 로그
- 해당 노드가 계속 스토리지 노드로 유지될 대상인지 여부

이 항목이 가장 중요합니다.

### 6.2 권장 해결 방향

#### 방안 A. 두 번째 SSD host 복구

가장 바람직한 방법입니다.

- `tempnode-bf3` 복구 또는
- 다른 SSD 노드 추가

장점:

- 현재 CRUSH/replica 정책 유지 가능
- 데이터 중복도 정상 회복 가능
- 구조적 일관성 유지

#### 방안 B. SSD pool 정책 재설계

만약 실제로 SSD host를 1개만 운영할 계획이라면, 현재 정책은 물리 현실과 맞지 않습니다.

가능한 조치:

- SSD 관련 pool의 replica 수 축소
- host 단위 CRUSH 배치 정책 완화

주의:

이 방법은 장애를 해결한다기보다, 낮은 중복도를 정책으로 인정하는 것입니다. 운영 리스크가 커서 예외적 선택으로 봐야 합니다.

### 6.3 추가 점검 필요 항목

#### 6.3.1 OSD provisioning 실패 원인 분석

다음 확인 필요:

- `rook-ceph-operator` 로그
- `rook-ceph-osd-prepare-*` job 로그
- device discovery 결과
- `CephCluster`의 storage node 정의

#### 6.3.2 최근 object store 설정 변경 검토

`dataPool` / `sharedPools.dataPool` 충돌이 있었으므로 관련 manifest나 values 변경 이력을 다시 확인해야 합니다.

#### 6.3.3 Probe 오류 원인 점검

다음 확인 필요:

- container runtime 상태
- 특정 노드에만 발생하는지 여부
- kubelet / runtime exec 오류 여부

## 7. 권장 확인 명령어

```bash
kubectl get cephcluster -n rook-ceph -o wide
kubectl describe cephcluster rook-ceph -n rook-ceph
kubectl get pods -n rook-ceph -o wide
kubectl get events -n rook-ceph --sort-by=.lastTimestamp
kubectl exec -n rook-ceph deploy/rook-ceph-tools -- ceph status
kubectl exec -n rook-ceph deploy/rook-ceph-tools -- ceph health detail
kubectl exec -n rook-ceph deploy/rook-ceph-tools -- ceph osd tree
kubectl exec -n rook-ceph deploy/rook-ceph-tools -- ceph osd pool ls detail
kubectl exec -n rook-ceph deploy/rook-ceph-tools -- ceph osd crush rule dump
kubectl exec -n rook-ceph deploy/rook-ceph-tools -- ceph pg dump_stuck undersized --format json
```

## 8. 결론

현재 Ceph 문제의 본질은 "용량 부족"이 아니라 "복제 배치 구조와 실제 SSD host 상태 불일치"입니다.

SSD 풀들은 host 단위 복제 2개를 요구하지만, 정상 SSD host가 사실상 1개뿐이므로 대량의 PG가 장기간 `undersized/degraded` 상태로 남아 있습니다.

따라서 가장 우선해야 할 일은:

1. `tempnode-bf3`의 SSD OSD 복구 여부 확인
2. 복구가 어렵다면 두 번째 SSD host 확보
3. 그것도 불가능하면 SSD pool 정책을 현재 토폴로지에 맞게 재설계

그 전까지는 클러스터를 정상 상태로 보기 어렵고, 특히 `sandbox-4`에 대한 운영 작업은 매우 신중해야 합니다.
