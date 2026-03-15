# Local Chart Patch Notes

## 대상
- 로컬 vendored chart 경로
  - `argocd/apps/06-DataX-WEKA/pydio/chart`
- 원본 기준
  - upstream `cells` chart `0.1.3`

## 수정 목적
- upstream chart 기본 install bootstrap 보완
- `Cloudflare Tunnel` 환경 대응
- `Ceph RGW` 기반 datasource bootstrap 제어
- MinIO 고정 bootstrap 제거

## 수정 파일
- `templates/configmap.yaml`
- `values.yaml`

## `templates/configmap.yaml` 수정
### `source` 블록
- 원본
  - `/vault/secrets/token` 무조건 읽기
- 수정
  - 파일 존재 시에만 `VAULT_TOKEN` export
- 이유
  - vault 비활성 또는 token 미주입 환경 대응
  - init 단계 불필요 오류 방지

### `proxyconfigs.reverseproxyurl`
- 원본
  - 값 사용 가능하나 실제 환경에서 미설정
- 수정
  - `service.reverseproxyurl` 사용
  - install-conf.yaml에 명시 렌더
- 이유
  - `pydio.jjooniex.org` host와 site 식별 일치
  - `Cannot detect site from incoming request` 방지

### admin bootstrap 계정
- 원본
  - `frontendlogin: admin`
  - `frontendpassword: P@ssw0rd`
  - chart 내부 하드코딩
- 수정
  - `.Values.install.frontendLogin`
  - `.Values.install.frontendPassword`
- 이유
  - values 기반 제어
  - install bootstrap 값 명시화

### datasource bootstrap
- 원본
  - `dstype: S3`
  - `dss3custom: {{ include "cells.minioURL" . }}`
  - `dss3apikey: {$MINIO_ROOT_USER}`
  - `dss3apisecret: {$MINIO_ROOT_PASSWORD}`
  - bucket 기본값
    - `pydiods1`
    - `personal`
    - `cellsdata`
    - `binaries`
    - `thumbnails`
    - `versions`
- 수정
  - `.Values.install.datasource.type`
  - `.Values.install.datasource.endpoint`
  - `.Values.install.datasource.accessKey`
  - `.Values.install.datasource.secretKey`
  - `.Values.install.datasource.bucketDefault`
  - `.Values.install.datasource.bucketPersonal`
  - `.Values.install.datasource.bucketCells`
  - `.Values.install.datasource.bucketBinaries`
  - `.Values.install.datasource.bucketThumbs`
  - `.Values.install.datasource.bucketVersions`
- 이유
  - MinIO 고정 bootstrap 제거
  - Ceph RGW endpoint 주입
  - OBC secret 기반 access key / secret key 사용 가능
  - datasource 생성 경로 values 제어 가능

## `values.yaml` 수정
### `install` 블록 추가
- 추가 키
  - `install.frontendLogin`
  - `install.frontendPassword`
  - `install.datasource.type`
  - `install.datasource.endpoint`
  - `install.datasource.accessKey`
  - `install.datasource.secretKey`
  - `install.datasource.bucketDefault`
  - `install.datasource.bucketPersonal`
  - `install.datasource.bucketCells`
  - `install.datasource.bucketBinaries`
  - `install.datasource.bucketThumbs`
  - `install.datasource.bucketVersions`
- 이유
  - 위 template 변경과 연결
  - install-conf 생성값을 values로 제어

## 변경 이유 요약
- upstream chart만으로 부족했던 항목
  - install 단계 datasource endpoint 제어
  - install 단계 S3 access key / secret key 제어
  - install 단계 bucket 이름 제어
  - reverse proxy URL 명시 보장
- 실제 운영 문제와 연결
  - `ReverseProxyURL` 불일치
  - `pydio-minio` bootstrap 고정
  - Ceph 연동 실패
  - login/session 초기화 실패

## 현재 이 patch가 필요한 이유
- 외부 진입점
  - `Cloudflare Tunnel`
- object storage
  - `rook-ceph-rgw-ceph-objectstore.rook-ceph.svc`
- 목표
  - Pydio install bootstrap 자체를 Ceph 기준으로 생성
  - runtime customconfigs 와 install bootstrap 간 불일치 제거

## 주의
- upstream chart 버전 업 시 재검토 필요
- 특히 확인 대상
  - `templates/configmap.yaml`
  - install bootstrap 관련 키 이름
  - datasource 생성 방식
  - reverse proxy 처리 방식
