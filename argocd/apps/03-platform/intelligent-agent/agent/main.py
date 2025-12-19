import os
import logging
import requests
import time
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from kubernetes import client, config

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SRE-Agent")

# --- [1] K8s 설정 (In-Cluster) ---
try:
    config.load_incluster_config()
    k8s_apps_v1 = client.AppsV1Api()
    k8s_core_v1 = client.CoreV1Api()
    logger.info("Kubernetes In-Cluster Config Loaded")
except Exception as e:
    logger.warning(f"Config 로드 실패 : {e}")

# --- [2] 모델 설정 (kubectl 결과 반영) ---

# GPU 1: 실행용 (Llama) - Service 이름: ollama-gpu1
# llama3:8b-instruct... -> llama3.1:8b 로 변경 (Tool Calling 공식 지원)
llm_llama = ChatOpenAI(
    base_url="http://ollama-gpu1.ollama.svc.cluster.local:11434/v1",
    api_key="ollama",
    model="llama3.1:8b",  
    temperature=0
)

# GPU 2: 실행용 (Mistral) - Service 이름: ollama-gpu2
# mistral:7b-instruct... -> mistral:v0.3 로 변경 (Tool Calling 공식 지원)
llm_mistral = ChatOpenAI(
    base_url="http://ollama-gpu2.ollama.svc.cluster.local:11434/v1",
    api_key="ollama",
    model="mistral:v0.3", 
    temperature=0
)

# --- [3] 도구(Tools) 정의 ---
@tool
def k8s_scale_deployment(namespace: str, deployment_name: str, replicas: int):
    """
    특정 Namespace의 Deployment Replica 수를 조정합니다.
    예: namespace='argocd', deployment_name='argocd-server', replicas=2
    """
    try:
        replicas = max(0, min(replicas, 5))  # 0~5 사이로 제한
        patch = {"spec": {"replicas": replicas}}
        k8s_apps_v1.patch_namespaced_deployment_scale(
            name=deployment_name,
            namespace=namespace,
            body=patch
        )
        return f"스케일링 완료: {namespace}/{deployment_name} -> {replicas}"
    except Exception as e:
        return f"스케일링 실패: {str(e)}"

@tool
def k8s_get_resources(namespace: str = "default"):
    """
    특정 네임스페이스의 파드 상태를 조회합니다.
    """
    try:
        pods = k8s_core_v1.list_namespaced_pod(namespace)
        status_list = [f"{p.metadata.name}: {p.status.phase}" for p in pods.items]
        return f"{namespace} 파드 목록:\n" + "\n".join(status_list[:10]) # 너무 길면 짤림 방지
    except Exception as e:
        return f"조회 실패: {str(e)}"

@tool
def get_prometheus_metrics(query_type: str):
    """
    클러스터의 상태 메트릭을 조회합니다.
    query_type에는 다음 중 하나를 입력하세요: 
    'cpu' (노드 CPU), 'memory' (노드 메모리), 'gpu' (GPU 사용률), 'unhealthy_pods' (비정상 파드)
    """
    
    # 1. 내부 DNS 주소 (values.yaml 기반)
    base_url = "http://prometheus-server.prometheus.svc.cluster.local"
    
    # 2. 질문 의도에 따른 PromQL 매핑
    queries = {
        "cpu": '100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)',
        "memory": '100 * (1 - ((node_memory_MemFree_bytes + node_memory_Buffers_bytes + node_memory_Cached_bytes) / node_memory_MemTotal_bytes))',
        "gpu": 'DCGM_FI_DEV_GPU_UTIL',
        "unhealthy_pods": 'kube_pod_status_phase{phase=~"Failed|Pending|Unknown"} > 0'
    }
    
    # 쿼리 선택 (없으면 그대로 실행 시도)
    promql = queries.get(query_type, query_type)
    
    try:
        # 3. 프로메테우스 API 호출
        response = requests.get(f"{base_url}/api/v1/query", params={'query': promql}, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        
        # 4. 결과 파싱 (읽기 좋게 변환)
        if data['status'] == 'success':
            results = data['data']['result']
            if not results:
                return "조회 결과 없음 (모두 정상일 수 있음)"
            
            output = []
            for res in results:
                # 결과에서 인스턴스 이름이나 Pod 이름 추출
                metric = res['metric']
                raw = res["value"][1]
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
                # 라벨에서 의미 있는 이름 찾기
                name = metric.get('instance') or metric.get('pod') or metric.get('gpu') or 'node'
                output.append(f"- {name}: {value:.2f}")
                
            return f"메트릭 조회 결과 ({query_type}):\n" + "\n".join(output)
        else:
            return f"프로메테우스 에러: {data['error']}"
            
    except Exception as e:
        return f"연결 실패: {str(e)}"

@tool
def ask_expert_analyst(context: str, question: str):
    """
    [Llama 호출] 복잡한 로그 분석이나 장애 원인 추론이 필요할 때 사용합니다.
    context에는 로그나 현재 상태를, question에는 구체적인 질문을 입력하세요.
    """
    logger.info(f"Expert(Llama) 호출: {question}")
    prompt = f"""
    Analyze the following situation and provide a concise root cause and solution.

    Context:
    {context}

    Question:
    {question}

    Act as a Senior SRE expert.
    """
    # GPU 0 (Llama) 모델 호출
    return llm_llama.invoke(prompt).content

# 도구 리스트 업데이트
tools = [k8s_scale_deployment, k8s_get_resources, ask_expert_analyst, get_prometheus_metrics]

# --- [4] 에이전트 생성 ---
system_prompt = """
You are 'SRE-Master', an intelligent Kubernetes Operations Agent.
Your responsibilities:
1. Direct Execution: Use tools like 'k8s_scale_deployment' or 'get_prometheus_metrics' for direct tasks.
2. Analysis: If the user asks for root cause analysis or "Why" something failed, YOU MUST USE 'ask_expert_analyst'.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm_mistral, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- [5] API 서버 ---
app = FastAPI()

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]

    class Config:
        extra = "ignore"

@app.post("/v1/chat/completions")
def chat_endpoint(req: ChatRequest):
    user_msg = req.messages[-1]['content']
    try:
        # invoke()는 Blocking 함수이므로 def 안에서 실행해야 안전함
        result = agent_executor.invoke({"input": user_msg})
        
        return {
            "choices": [{
                "message": {"role": "assistant", "content": result['output']}
            }]
        }
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "sre-agent",  
                "object": "model",
                "created": int(time.time()),
                "owned_by": "sre-team",
            }
        ]
    }