# Ceph 클러스터 상태 메모

- 작성일: 2026-03-11
- 대상: `rook-ceph`
- 클러스터 ID: `31d21ce2-d155-4c3e-8801-b14943c8b73d`

## 현재 상태

- `HEALTH_WARN`
- `36 osds: 36 up / 36 in`
- `mon quorum: c,d,e`
- `mgr: a active, b standby`
- `mds: 1 active, 1 standby`
- `rgw: 1 active`
- `362497/730423 objects degraded (49.628%)`
- `48 pgs degraded`
- `52 pgs undersized`
- `55 pgs not deep-scrubbed in time`
- `55 pgs not scrubbed in time`
- `recovery 진행 중`
- `recovery: 117 MiB/s, 30 objects/s`

## 핵심 판단

- 클러스터 자체 down 상태 아님
- monitor, mgr, mds, rgw 정상 복귀 상태
- OSD 수량 기준 정상 복귀 상태
- 데이터 중복도 저하 문제는 아직 남은 상태
- 기존 장기 undersized PG와 신규 backfill/recovery 동시 진행 상태

## 장애 배경

- SSD pool 다수
- replica `size 2`
- CRUSH rule `chooseleaf type host`
- host 단위 분산 전제
- 한동안 `sandbox-4` 중심 단일 SSD host 의존 상태
- `tempnode-bf3`의 `osd.2`, `osd.4` 이탈 상태
- 결과: SSD 계층 PG `undersized`, `degraded` 장기 고착

## 이번 점검에서 확인한 문제 축

### 1. `tempnode-bf3` OSD 미복귀

- 과거 수동 제외 이력
- node taint 존재
- `doca=dedicated:NoSchedule`
- `gpu=dedicated:NoSchedule`
- 초기 상태에서 Rook placement 불일치
- `role=storage-node` 라벨 부재
- toleration 부재

### 2. `sandbox-4` Ceph pod 기동 실패

- `mon.e`, 다수 OSD pod `Init/Pending`
- 공통 이벤트: `FailedCreatePodSandBox`
- 원인: `multus-shim` 호출 실패
- `multus-cni-node` OOMKilled 반복
- 이후 `multus-shim` 바이너리 교체 중 `Text file busy`

### 3. `tempnode-bf3` 기존 OSD 메타데이터 문제

- prepare 단계 자체는 성공
- `osd.2 -> /dev/sdb`
- `osd.4 -> /dev/sdc`
- deployment 생성 이후 `expand-bluefs` init crash
- `BlueStore::expand_devices()` assert
- `not all labels read properly`
- `osd init failed: (5) Input/output error`

## 진행한 조치

### A. Rook placement / toleration 정리

- `tempnode-bf3`에 `role=storage-node` 라벨 반영
- `rook-ceph-cluster` values 수정
- `placement.all` toleration 추가
- `placement.osd` toleration 추가
- `placement.prepareosd` toleration 추가
- 추가 항목
- `doca`
- `gpu`
- 결과
- `tempnode-bf3` 대상 prepare job 생성 가능 상태

### B. Multus 복구

- `multus-cni-node` OOM 원인 확인
- multus values 조정
- `logLevel: debug -> info`
- resources 상향
- 실패 중인 multus pod 재기동
- `sandbox-4` multus 정상화
- 결과
- `mon.e` 복구
- `sandbox-4` OSD 다수 복구
- Ceph pod sandbox 생성 문제 해소

### C. `osd.2`, `osd.4` 복구 시도

- prepare job 재실행
- raw device 인식 확인
- `osd.2` / `osd.4` deployment 생성 확인
- `expand-bluefs` 단계 crash 확인
- 임시로 `expand-bluefs` init 제거 테스트
- OSD 본체 컨테이너까지 진입 확인
- 최종적으로 object store mount 실패 확인
- 판단
- 기존 BlueStore 메타데이터 복구보다 재생성 쪽이 현실적

### D. `osd.2`, `osd.4` 폐기 및 재생성 준비

- `ceph osd purge 2 --yes-i-really-mean-it`
- `ceph osd purge 4 --yes-i-really-mean-it`
- `rook-ceph-osd-2`, `rook-ceph-osd-4` deployment 삭제
- `tempnode-bf3`에서 디스크 정리
- `sgdisk --zap-all`
- `wipefs -af`
- `blkdiscard`
- OSD host path 정리
- 추가 확인
- `ceph-volume raw list /host/dev/sdb -> {}`
- `ceph-volume raw list /host/dev/sdc -> {}`
- 의미
- 현재 디스크 raw metadata 제거 상태

### E. operator 재기동

- `rook-ceph-operator` pod 재시작
- 전체 reconcile 재유도
- 결과
- cluster reconcile 재개
- `tempnode-bf3` host 재편입
- 현재 `ceph osd tree` 기준 `osd.2`, `osd.4` 복귀 상태

## 현재 결과

- `tempnode-bf3` 아래 SSD OSD 2개 복귀
- `osd.2 up`
- `osd.4 up`
- 전체 OSD `36 up / 36 in`
- `sandbox-4` 단일 host 의존 완화
- host 단위 SSD replica 배치 조건 일부 회복
- degraded/undersized PG 즉시 전부 해소 상태는 아님
- recovery/backfill 진행 중

## 아직 남은 문제

- `PG_DEGRADED` 잔존
- `PG undersized` 잔존
- scrub / deep-scrub 지연 잔존
- 일부 PG
- 수 주 단위 장기 stuck 이력
- 일부 PG
- 최근 remapped/backfill 상태

## 현재 해석

- 구조적 대형 장애 구간은 벗어난 상태
- `tempnode-bf3` OSD 복귀로 토폴로지 정상화 방향
- 기존 장기 누적 문제 + 재복구 직후 backfill 영향으로 `HEALTH_WARN` 유지 상태
- 단기적으로 recovery 숫자 변동 가능
- 일정 시간 경과 후 `undersized`, `degraded` 수치 추가 감소 기대

## 운영상 주의

- recovery 완료 전 불필요한 OSD 재시작 지양
- `sandbox-4`, `tempnode-bf3` 재부팅 지양
- 추가 disk wipe / purge 작업 중지
- operator가 다시 `osd.2`, `osd.4` 교체 대상으로 보지 않는지 관찰 필요

## 다음 확인 포인트

- `ceph status`
- `ceph health detail`
- `ceph osd tree`
- `ceph pg stat`
- `ceph pg dump_stuck undersized`
- `kubectl get pods -n rook-ceph -o wide`
- `kubectl logs -n rook-ceph deploy/rook-ceph-operator`

## 다음 목표 상태

- `36 up / 36 in` 유지
- `tempnode-bf3` OSD 안정 유지
- `degraded` PG 추가 감소
- `undersized` PG 추가 감소
- scrub / deep-scrub 경고 감소
- 최종적으로 `HEALTH_OK` 또는 최소 `HEALTH_WARN` 경고 항목 대폭 축소

## 2026-03-12 추가 점검

### 1. 현재 확인한 핵심 상태

- 기준 시각: `2026-03-12 UTC`
- `ceph status`
- `36 osds up / 36 in`
- `19 pools`, `99 pgs`
- `HEALTH_WARN` 유지
- 남은 경고
- `26 pgs degraded`
- `31 pgs undersized`
- `14 active+clean+remapped`
- `30 pgs not scrubbed in time`
- `30 pgs not deep-scrubbed in time`
- 추가 경고
- `mon.e` clock skew 약 `0.059s`

### 2. RGW / sharedPools 재확인 결과

- `CephObjectStore ceph-objectstore` 선언은 이미 `sharedPools` 기준으로 적용되어 있음
- 실제 운영 realm/zone도 `ceph-objectstore` 사용 중
- `bucket list`, `user list`, `ObjectBucketClaim`, `CephObjectStoreUser` 모두 정상 확인
- bucket placement
- `STANDARD -> rgw-hot-pool-ssd`
- `STANDARD_IA -> rgw-cold-pool-hdd`
- 따라서 이번 시점의 주 장애 원인은 `sharedPools` 미적용이 아님
- `default.rgw.*`, `ceph-objectstore.rgw.*` 일부 잔여 풀은 legacy 흔적으로 보이며, 현재 운영 경로의 직접 원인으로 판단하지 않음

### 3. Rook disruption 상태 추가 확인

- 초기 추적 시 `tempnode-bf3`가 문제처럼 보였으나, 추가 확인 결과 실제 stale drain state는 `sandbox-4`로 확인됨
- 근거
- `rook-ceph-pdbstatemap`에 `draining-failure-domain: sandbox-4` 저장
- `sandbox-4` 노드는 실제로 `Ready`
- cordon 상태 아님
- 해당 stale state 때문에 `clusterdisruption-controller`가 반복적으로
- `Draining Failure Domain: "sandbox-4"`
- `OSD ... is not ok-to-stop`
- 로그를 계속 남기며 OSD reconcile을 지연시키는 상태였음

### 4. 2026-03-12 수행한 조치

- live cluster에서 `rook-ceph-pdbstatemap`의 drain 상태 초기화
- 비운 값
- `draining-failure-domain`
- `draining-failure-domain-duration`
- `set-no-out`

### 5. 조치 직후 확인된 결과

- Rook operator 로그에서 아래 전환 확인
- `all PGs are active+clean. Restoring default OSD pdb settings`
- 임시 host PDB 삭제
- 기본 PDB `rook-ceph-osd` 복구
- 실제 PDB 상태도 임시 host PDB 제거 후 `rook-ceph-osd`만 남는 것으로 확인
- 의미
- 적어도 stale drain / PDB 루프는 해소됨
- `sandbox-4`를 계속 drain 중인 failure domain으로 잘못 붙잡고 있던 상태는 정리됨

### 6. 조치 후에도 남아 있는 상태

- `ceph status` 자체는 즉시 정상화되지 않음
- 여전히 `26 pgs degraded`, `31 pgs undersized` 유지
- `CephCluster` 상태도 여전히
- `Processing OSD 34 on node "sandbox-4"`
- `Phase=Progressing`
- `State=Creating`
- 즉
- `stale drain state`는 보조 문제였고
- 실제 본체 문제는 장기 누적된 `undersized/degraded PG` 상태가 아직 남아 있다는 뜻

### 7. 현재 해석

- `sharedPools`는 정상
- `tempnode-bf3`는 현재 `Ready`, `unschedulable=false`, OSD `2`, `4`도 `up/in`
- 현재 남은 경고의 중심은 SSD 계열 2-host 구조에서 한동안 발생한 복제 저하 PG의 장기 잔존 상태
- `size: 2`, `failureDomain: host` 자체는 현재 의도한 설정과 일치하지만, SSD host가 정확히 2개뿐이라 둘 중 하나가 흔들리면 바로 `undersized`가 발생함
- 이번 점검으로 확인된 것은
- 잘못 남아 있던 drain 보호 상태는 정리 가능
- 그러나 장기 `undersized` PG는 별도로 추적해야 함

### 8. 앞으로 어떻게 할지

- 1단계
- `ceph status`
- `ceph health detail`
- `kubectl get cephcluster rook-ceph -n rook-ceph`
- 위 3가지를 기준으로 `Processing OSD 34`가 끝나는지 먼저 관찰
- 2단계
- `mon.e` clock skew 해소
- NTP/chrony 동기화 확인
- 이는 주원인은 아니지만 health noise를 줄이는 데 필요
- 3단계
- 남은 `undersized` PG를 pool 단위로 분해해서
- 어떤 pool이 아직도 회복을 못 하는지 확인
- 특히 `hot-pool-ssd`, `ceph-filesystem-hot-*`, `rgw-hot-pool-ssd` 계열 우선 확인
- 4단계
- recovery가 더 진행되지 않고 동일 PG가 계속 고정되면
- 해당 pool의 CRUSH 배치와 acting set을 기준으로 개별 원인을 다시 확인
- 이때는 `PG -> pool -> OSD -> host` 순서로 추적

### 9. 현 시점 운영 권고

- 지금은 추가 purge / wipe / OSD 재생성 작업을 바로 반복하지 않는 것이 맞음
- 먼저 stale drain 루프 해소 후 Ceph가 스스로 얼마나 회복하는지 봐야 함
- 즉시 권장하는 운영 순서
- `Processing OSD 34` 종료 여부 확인
- `mon.e` 시간 동기화 확인
- 10~30분 간격으로 `ceph status` 재확인
- 그래도 `undersized` PG가 그대로 고정이면, 그때 pool/PG 기준 심화 분석으로 이동
