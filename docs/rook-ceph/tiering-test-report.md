# [실험 보고서] Ceph S3 스토리지 티어링(Tiering) 검증

## 1. 실험 방법 (Experimental Method)
- **테스트 환경**: OCIS 전용 S3 버킷(`ocis`) 및 RGW 계층화 풀 설정 완료 상태
- **테스트 데이터**: 1MB 크기의 임의 데이터 파일 (`tiering-test.dat`)
- **수행 절차**:
  1. `mc` (MinIO Client)를 이용해 파일을 기본 클래스(`STANDARD`)로 업로드하여 SSD 풀(`rgw-hot-pool-ssd`) 진입 확인
  2. `radosgw-admin object stat` 명령어로 객체의 물리적 배치 규칙(`placement_rule`) 및 스토리지 클래스 확인
  3. `mc cp --storage-class STANDARD_IA` 명령어를 사용하여 동일 객체의 클래스를 HDD 풀용 클래스로 업데이트
  4. 다시 `object stat`을 조회하여 물리적 풀 이동 및 클래스 변경 여부 최종 검증

## 2. 실험 결과 (Results)
- **초기 상태 (SSD)**:
  - **Storage Class**: `STANDARD` (Default)
  - **Placement Rule**: `default`
  - **실제 저장 위치**: `rgw-hot-pool-ssd` 풀
- **전환 후 상태 (HDD)**:
  - **Storage Class**: `STANDARD_IA` (변경 완료)
  - **Placement Rule**: `default/STANDARD_IA` (변경 완료)
  - **실제 저장 위치**: `rgw-cold-pool-hdd` 풀 매핑 확인
- **검증 결론**: S3 API 레벨의 명령어 호출만으로 데이터가 고성능 SSD 풀에서 고용량 HDD 풀로 물리적으로 성공적으로 이관됨을 확인함

## 3. 통찰 (Insights)
- **추상화 성공**: 어플리케이션(OCIS)이나 사용자 입장에서는 파일의 경로(URL) 변경 없이 내부 속성(Storage Class) 변경만으로 물리적인 저장 매체를 전환할 수 있음
- **유연한 매핑**: Ceph RGW의 `sharedPools`와 `placement_pools` 설정을 통해 논리적인 S3 클래스와 물리적인 Rados 풀 간의 복잡한 매핑 관계를 안정적으로 제어 가능함
- **성능/비용 최적화**: 모든 데이터를 SSD에 담지 않고, 필요에 따라 HDD로 밀어낼 수 있는 인프라 배관이 완벽히 구축됨을 입증함

## 4. 향후 활용 방안 (Future Utilization)
- **자동 라이프사이클 정책 (LC)**: `ocis` 버킷에 RGW Lifecycle Rule을 적용하여, 업로드 후 30일이 경과한 파일은 자동으로 `STANDARD_IA`(HDD)로 강등(Transition)되도록 설정
- **태그 기반 티어링**: 특정 프로젝트나 사용자의 데이터에 'Cold' 태그를 부여하고, 이를 감지하여 HDD 풀로 옮기는 자동화 스크립트(Sidecar 등) 연동
- **공간 효율화**: OCIS의 'Spaces' 개념과 연동하여, 중요도가 낮은 아카이브용 스페이스는 생성 시점부터 HDD 풀을 바라보도록 설정 유도
