# Harbor 사용 및 트러블슈팅 가이드

## 1. 퀵 스타트 (사용법)
* **Harbor UI 접속**: `http://10.34.25.12` (ID: `admin` / PW: `test1234`)
* **외부 노드 Docker 로그인**: 
  ```bash
  # LoadBalancer 설정에 따라 80 포트 명시 필수
  sudo docker login 10.34.25.12:80 -u admin -p test1234
  ```
* **이미지 푸시 (Push)**:
  ```bash
  docker tag <image>:<tag> 10.34.25.12:80/<project>/<image>:<tag>
  docker push 10.34.25.12:80/<project>/<image>:<tag>
  ```
* **K8s Image Pull Secret 생성**:
  ```bash
  kubectl create secret docker-registry harbor-pull-secret \
    --docker-server=10.34.25.12:80 \
    --docker-username=admin \
    --docker-password=test1234 \
    --namespace <대상-네임스페이스>
  ```

## 2. 이슈 및 해결 내역

### 이슈 A: Docker HTTPS 강제 접속 문제
* **현상**: Harbor가 HTTP(80)임에도 Docker가 HTTPS(443) 접속을 시도하여 타임아웃 발생
* **원인**: Docker 데몬의 기본 보안 정책 (Insecure Registry 미등록 시 HTTPS 강제)
* **해결**: 
  * 외부 노드의 `/etc/docker/daemon.json` 수정
  * `"insecure-registries": ["10.34.25.12", "10.34.25.12:80"]` 추가
  * Docker 서비스 재시작 (`systemctl restart docker`)

### 이슈 B: Docker CLI 로그인 401 Unauthorized 에러
* **현상**: UI 로그인은 정상이지만, Docker CLI 로그인 시에만 401 에러 발생
* **원인**: `harbor-core`(토큰 발행)와 `harbor-registry`(토큰 검증) 간의 내부 인증 키(Secret Key) 불일치
* **상세 해결 과정**:
  1. **내부 시크릿 동기화**: 
     - Helm 배포 시마다 랜덤 생성되던 내부 키를 `harbor-secretkey`로 고정
     - `harbor-secretkey-secret.yaml`에 `secret`, `JOBSERVICE_SECRET`, `REGISTRY_HTTP_SECRET` 키 강제 정의
  2. **Registry 인증 로직 수정**:
     - `values.yaml`에서 `registry.credentials.existingSecret` 제거
     - 레지스트리가 자체 DB 계정 대신 Core가 발행한 토큰 인증만 따르도록 강제
  3. **경로 처리 개선**:
     - `registry.relativeurls: true` 설정으로 LoadBalancer 뒤에서의 절대 경로 리다이렉트 이슈 방지
  4. **External URL 정렬**:
     - `externalURL: http://10.34.25.12`로 설정하여 토큰 내 Audience 정보 일치화

## 3. 설정 유지보수 및 참고사항
* **배포 전략**: ArgoCD Sync Wave(29-30)를 통한 순차 배포 관리
* **시크릿 보안**: Harbor 내부 통신용 키는 16자 이상 필수 (현재 `test1234test1234` 사용)
* **저장소**: `rook-ceph-block-hot` (RWO) 기반 영구 볼륨 사용
