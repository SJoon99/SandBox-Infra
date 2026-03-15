# Pydio 로그인 장애 정리

## 장애 시점
- 대상 앱: `argocd/apps/06-DataX-WEKA/pydio`
- 확인 일자: `2026-03-15`
- 외부 노출: `Cloudflare Tunnel`
- 접속 URL: `https://pydio.jjooniex.org`

## 실제 증상
- 로그인 화면 노출
- 로그인 요청 실패
- 세션 생성 실패
- 인증 초기화 panic
- datasource sync 경고 반복

## 핵심 로그
- `Cannot detect site from incoming request`
- `ReverseProxyURL ... should match the incoming host 'pydio.jjooniex.org'`
- `[REST]/a/frontend/session rpc error`
- `Cannot contact s3 service`
- `pydio-minio.pydio.svc.cluster.local:9000`
- `dial tcp: lookup ... no such host`

## 실제 원인
### 1. 인증 경로 불일치
- 외부 접속 호스트: `pydio.jjooniex.org`
- chart 기본 install bootstrap:
  - `reverseproxyurl` 미설정
- 결과:
  - OAuth issuer 계산 실패
  - site resolution 실패
  - 로그인 요청 시 panic

근거:
- 현재 수정값: [values.yaml](/home/joon/SandBox-Infra/argocd/apps/06-DataX-WEKA/pydio/values.yaml#L39)
- 현재 템플릿 반영 위치: [configmap.yaml](/home/joon/SandBox-Infra/argocd/apps/06-DataX-WEKA/pydio/chart/templates/configmap.yaml#L10)

### 2. install bootstrap datasource 오염
- 의도한 storage:
  - Ceph RGW
  - `rook-ceph-rgw-ceph-objectstore.rook-ceph.svc`
- 실제 chart 0.1.3 기본 install-conf:
  - `dss3custom: {{ include "cells.minioURL" . }}`
  - bucket 기본값: `pydiods1`, `personal`, `cellsdata`, `binaries`, `thumbnails`, `versions`
- 클러스터 상태:
  - `minio.enabled: false`
  - `pydio-minio` 서비스 없음
- 결과:
  - 첫 bootstrap 시 datasource metadata가 MinIO 기준으로 생성
  - 런타임에서 계속 `pydio-minio` 조회
  - sync warning 반복

근거:
- 원본 문제 템플릿 위치: [configmap.yaml](/home/joon/SandBox-Infra/argocd/apps/06-DataX-WEKA/pydio/chart/templates/configmap.yaml#L53)
- Ceph 의도값 위치: [values.yaml](/home/joon/SandBox-Infra/argocd/apps/06-DataX-WEKA/pydio/values.yaml#L45)

### 3. 상태 저장소에 잘못된 bootstrap 값 영구화
- 설치 init container 동작:
  - `cells configure`
  - install-conf.yaml 읽기
  - etcd / db에 초기 설정 기록
- 기록 대상:
  - etcd config
  - MariaDB
  - MongoDB
- 특징:
  - bootstrap 1회성
  - 잘못된 초기값도 정상 상태처럼 저장
  - 이후 values 수정만으로 완전 복구 불가 가능성
- 의미:
  - 이미 생성된 site / datasource / auth 관련 메타데이터 재사용
  - 로그인 실패 원인 지속
  - MinIO endpoint 흔적 지속

## 왜 외부 chart 대신 로컬 chart 사용
### 외부 chart 유지 시 장점
- 업스트림 추적 단순
- 저장소 구조 간결
- 차트 버전 업 명확

### 이번 케이스에서 외부 chart 한계
- chart `0.1.3`의 `install-conf.yaml` 생성 로직 고정
- `reverseproxyurl` 지원:
  - values로 가능
- datasource bootstrap override:
  - values만으로 불충분
  - `dss3custom`, bucket 명, admin 자격증명 주입 경로 부재
- 즉:
  - `customconfigs.defaults/external/url`만 수정 가능
  - install bootstrap 본문 자체는 MinIO 기준 유지

### 로컬 chart 선택 이유
- 최소 범위 patch
- bootstrap template 직접 제어
- `Cloudflare Tunnel + Ceph` 조합 명시화
- 업스트림 chart 전체 재작성 아님
- 필요한 부분만 수정:
  - `source` script guard
  - `reverseproxyurl`
  - admin credential 변수화
  - datasource endpoint/bucket 변수화

### 결론
- 일반론:
  - 원격 chart 선호
- 이번 건:
  - bootstrap template 제어 필요
  - 원격 chart만으로 요구사항 충족 불가
  - 로컬 vendoring 불가피

## 왜 기존에 로그인이 안 되었는지
- 외부 URL: `https://pydio.jjooniex.org`
- Tunnel 경유 요청 Host:
  - `pydio.jjooniex.org`
- Pydio bootstrap 상태:
  - site reverse proxy URL 미정합
  - auth provider 초기화 실패
- 동시 발생:
  - datasource bootstrap MinIO 오염
  - background sync warning 대량 발생
- 최종 결과:
  - `/a/frontend/session` 실패
  - 로그인 불가

## 해결 방향
### 필수 수정
- `service.reverseproxyurl: https://pydio.jjooniex.org`
- `customconfigs.defaults/external/url: https://pydio.jjooniex.org`
- install datasource endpoint:
  - `http://rook-ceph-rgw-ceph-objectstore.rook-ceph.svc`
- install bucket 값:
  - Ceph 기준으로 일치화
- Pydio service:
  - `ClusterIP`
- 외부 진입:
  - `Cloudflare Tunnel` 유지

### 상태 초기화 필요 이유
- 이미 잘못 생성된 bootstrap state 존재 가능성
- 새 values 반영만으로 기존 site/datasource 재생성 보장 없음
- 안전한 복구 방향:
  - Pydio app sync
  - Pydio config/state reset
  - fresh bootstrap

## 구현 반영 내용
- Argo CD app source:
  - 원격 chart -> 로컬 vendored chart
- 로컬 chart patch:
  - install-conf datasource override 지원
  - reverseproxyurl 렌더링
  - vault token file guard
- values 정렬:
  - `ClusterIP`
  - `reverseproxyurl`
  - `install.datasource.*`

## 검증 포인트
- 렌더 결과 내 `pydio-minio` 문자열 부재
- 렌더 결과 내 `reverseproxyurl: "https://pydio.jjooniex.org"`
- 렌더 결과 내 `dss3custom: "http://rook-ceph-rgw-ceph-objectstore.rook-ceph.svc"`
- 재배포 후 로그:
  - `Cannot detect site from incoming request` 소멸
  - `Cannot contact s3 service` 소멸
- 브라우저 테스트:
  - 로그인 성공
  - 세션 생성 성공
  - 파일 업로드 성공

## 남은 작업
- Git 반영
- Argo CD sync
- 기존 Pydio state reset 범위 확정
- Cloudflare Tunnel origin 설정 재확인
  - origin host
  - Host header 유지
  - `X-Forwarded-Proto: https`
