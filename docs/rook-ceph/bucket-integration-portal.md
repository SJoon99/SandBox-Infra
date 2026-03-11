# [통합 보고서] OCIS-Ceph S3 버킷 연동 포털

## 1. 개요
- **작업 목적**: OCIS 데이터 백엔드 S3 전환 및 스토리지 티어링(SSD/HDD) 기반 마련
- **최종 상태**: 연동 완료, 실데이터 업로드 검증 성공
- **핵심 정보**:
  - **버킷명**: `ocis`
  - **관리자ID**: `admin`
  - **신규PW**: `kSWOmhywkHU5oAOrC1ddgUhlsrVgPu`

## 2. 장애 복구 및 연동 일지 (2026-03-11)

### [Issue 01] S3 드라이버 미작동
- **현상**: 설정 주입 후에도 로컬 스토리지에 데이터 저장됨
- **원인**: Helm `values.yaml` 내 `storageBackend` 설정 계층 구조 누락
- **조치**: `services.storageusers.storageBackend` 하위로 `driver: s3ng` 정상 중첩 설정

### [Issue 02] 파드 기동 정체 (Multi-Attach)
- **현상**: `storageusers` 파드가 볼륨 점유 에러로 `ContainerCreating` 상태 지속
- **원인**: RWO 볼륨 환경에서 `RollingUpdate` 전략 사용으로 인한 자원 충돌
- **조치**: 배포 전략을 `Recreate`로 변경하여 볼륨 완전 해제 후 기동 유도

### [Issue 03] 오브젝트 스토어 수렴 실패
- **현상**: OBC(ObjectBucketClaim)를 통한 유저/버킷 생성 안 됨
- **원인**: `CephObjectStore` 내 `dataPool`과 `sharedPools` 중복 정의 (상호 배타적 설정)
- **조치**: 중복 `dataPool` 제거 및 계층화(SSD/HDD) 지원 `sharedPools` 설정으로 단일화

### [Issue 04] RGW 유저/키 불일치
- **현상**: 파드에 주입된 S3 Key로 RGW 접속 실패 (403 Forbidden)
- **원인**: Ceph 클러스터 불안정으로 오퍼레이터 동기화 지연 및 OBC 재생성 시 키 변경
- **조치**: `radosgw-admin`으로 유저 수동 생성 및 OCIS Secret 키값과 강제 동기화

### [Issue 05] 서비스 간 인증 거부 (JWT Mismatch)
- **현상**: 로그인 불가 및 `signature is invalid` 에러 발생
- **원인**: 수동 JWT Secret 참조 오류 및 서비스 간 인증 키 불일치
- **조치**: Secret 참조 원복(자동 생성) 및 `ocis` 네임스페이스 전체 재시작으로 키 전파

## 3. 스토리지 계층 구조 (Tiering Map)

| 스토리지 클래스 | 물리 풀 (Ceph Pool) | 하드웨어 타입 | 비고 |
| :--- | :--- | :--- | :--- |
| **STANDARD** | `rgw-hot-pool-ssd` | **SSD** | OCIS 기본 저장소 |
| **STANDARD_IA** | `rgw-cold-pool-hdd` | **HDD** | 저비용/장기 보관용 |

## 4. 검증 결과
- **S3 연결성**: `mc` (MinIO Client)를 통한 `ocis` 버킷 읽기/쓰기 성공
- **데이터 현황**: RGW `ceph-objectstore` 존 내 객체 121개 존재 확인
- **웹 서비스**: 신규 관리자 비밀번호 기반 로그인 및 UI 정상 작동 확인
