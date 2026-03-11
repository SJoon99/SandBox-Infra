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
