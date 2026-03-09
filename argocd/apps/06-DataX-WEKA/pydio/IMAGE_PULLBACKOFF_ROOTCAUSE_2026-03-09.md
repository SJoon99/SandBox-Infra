# Pydio ImagePullBackOff 장애 보고서

## 문서 메타
- 작성일: 2026-03-09 (UTC)
- 대상 앱: ArgoCD `pydio` (chart: `cells`, version: `0.1.3`)
- 대상 파일: `argocd/apps/06-DataX-WEKA/pydio/values.yaml`
- 장애 유형: `ErrImagePull` / `ImagePullBackOff`

## 장애 증상
- 실패 이미지: `docker.io/bitnami/os-shell:12-debian-12-r43`
- 이벤트 핵심 메시지:
  - `Back-off pulling image`
  - `unexpected media type text/html`
  - `failed to unpack image ... overlayfs`
- 영향 범위:
  - Bitnami 의존 서브차트 init 컨테이너(`volumePermissions`) 기동 실패
  - 파드 초기화 지연/실패

## 원인 분석
- 1차 원인: Docker Hub 경로(`docker.io`) pull 경로 문제
  - Rate Limit 초과 가능성
  - 사내망/프록시/방화벽 개입 가능성
  - 특정 태그 응답 비정상(이미지 레이어 대신 HTML 에러 페이지)
- 2차 원인: Bitnami 차트 이미지 검증 정책
  - 비표준 레지스트리 사용 시 템플릿/배포 차단 가능
  - `global.security.allowInsecureImages` 미설정 시 검증 실패

## 기술적 근거
- `unexpected media type text/html` 의미:
  - OCI 레이어 expected
  - 실제 응답: HTML 문서(에러/안내 페이지)
- 해석:
  - 컨테이너 런타임 관점 정상 이미지 응답 아님
  - 네트워크/레지스트리 응답 경로 이상 징후

## 해결 전략
- 전략 A: 레지스트리 우회
  - from: `docker.io/bitnami/*`
  - to: `public.ecr.aws/bitnami/*` (AWS Public ECR 미러)
- 전략 B: Bitnami 이미지 검증 정책 허용
  - `global.security.allowInsecureImages: true`

## 실제 변경 사항
- 변경 파일: `argocd/apps/06-DataX-WEKA/pydio/values.yaml`
- 추가 블록:

```yaml
global:
  imageRegistry: public.ecr.aws
  security:
    allowInsecureImages: true
```

## 변경 효과
- Bitnami 의존 이미지 경로 전환:
  - `docker.io/bitnami/os-shell:12-debian-12-r43`
  - `public.ecr.aws/bitnami/os-shell:12-debian-12-r43`
- 동일 패턴 적용 대상:
  - `mariadb`, `redis`, `nats`, `mongodb`, `etcd` 등 Bitnami 서브차트 이미지

## 검증 결과
- Helm 렌더링 기준 확인:
  - `public.ecr.aws/bitnami/os-shell:*` 출력 확인
  - `public.ecr.aws/bitnami/mariadb:*` 출력 확인
  - `public.ecr.aws/bitnami/redis:*` 출력 확인
- 결과: 이미지 경로 전환 정상 반영

## 운영 반영 절차
- 1단계: Git 반영
- 2단계: ArgoCD `pydio` 앱 Sync
- 3단계: 파드 상태 확인
  - `kubectl -n pydio get pods`
  - `kubectl -n pydio describe pod <pod-name>`
- 4단계: 이미지 Pull 이벤트 재확인
  - `kubectl -n pydio get events --sort-by=.lastTimestamp`

## 재발 방지
- 레지스트리 정책 표준화:
  - Bitnami 의존 차트 기본 레지스트리 통일(`public.ecr.aws`)
- 사전 검증 루틴:
  - `helm template` 렌더링 시 이미지 경로 grep 체크
  - 운영 노드 `ctr images pull` 사전 테스트
- 모니터링:
  - `ErrImagePull`, `ImagePullBackOff` 이벤트 알림 룰

## 롤백 포인트
- 즉시 롤백 키:
  - `global.imageRegistry` 제거 또는 빈 값
  - `global.security.allowInsecureImages` 제거/false
- 롤백 영향:
  - Docker Hub 경로 복귀
  - 기존 Rate Limit/네트워크 이슈 재노출 가능성

## 참고
- 장애 발생 시간대(사용자 보고): 2026-03-09 10:22 ~ 10:25 (로컬 표기)
- 핵심 에러 키워드: `unexpected media type text/html`, `ImagePullBackOff`, `ErrImagePull`
