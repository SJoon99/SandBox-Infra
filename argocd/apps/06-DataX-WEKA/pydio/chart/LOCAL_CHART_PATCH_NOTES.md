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

## 원격 Chart 배포 시 이슈 정리
### 이슈 1. `helm dependency build` 실패
- 실행 명령
  - `helm dependency build ./argocd/apps/06-DataX-WEKA/pydio/chart`
- 실제 에러
  - `Error: no repository definition for https://kubernetes.github.io/ingress-nginx, https://helm.releases.hashicorp.com, https://charts.jetstack.io. Please add the missing repos via 'helm repo add'`
- 원인
  - `Chart.yaml` / `Chart.lock` 에 선언된 dependency repository 가 로컬 Helm repo 설정에 없어서 발생
  - 특히 누락된 repo
    - `https://kubernetes.github.io/ingress-nginx`
    - `https://helm.releases.hashicorp.com`
    - `https://charts.jetstack.io`
  - `bitnami` repo도 환경에 따라 없을 수 있으므로 같이 추가하는 편이 안전

### 이슈 1의 의미
- 이 에러는 chart 자체가 깨졌다는 뜻은 아님
- 현재 `chart/charts/` 아래에 dependency `.tgz` 파일이 이미 vendoring 되어 있으면
  - `helm package` 는 바로 성공할 수 있음
  - 실제 확인 결과 `/tmp/cells-0.1.3.tgz` 패키징 성공
- 즉
  - dependency 를 "다시 받아서 재구성" 하려면 `helm repo add` 가 필요
  - 현재 vendored dependency 그대로 "패키징만" 하려면 바로 진행 가능

### 필요한 Helm repo 등록 명령
```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo add hashicorp https://helm.releases.hashicorp.com
helm repo add jetstack https://charts.jetstack.io
helm repo update
```

### 권장 절차
1. dependency 를 최신 lock 기준으로 다시 맞출 필요가 있으면 `helm repo add` 후 `helm dependency build`
2. vendored dependency 를 그대로 사용할 거면 `helm package ./argocd/apps/06-DataX-WEKA/pydio/chart`
3. OCI registry 배포 시 `helm push` 수행

### 현재 판단
- 이번 작업에서는 `chart/charts/*.tgz` 가 이미 존재하므로
  - 급한 목적이 "원격 chart 로 전환" 이라면 `helm package` -> `helm push` 로 먼저 진행 가능
- 다만 장기적으로는
  - repo 등록 후 `helm dependency build`
  - 필요 시 `Chart.lock` 재생성 여부 검토
  - 그 다음 package/push
  순서가 더 재현 가능성이 높음
