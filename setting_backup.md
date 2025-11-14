# 📄 WSL2 GPU Whisper 환경 구성 명세서 (2025-11-14 기준)

WSL2 환경에서 GPU 기반 Whisper(faster-whisper + CTranslate2)를
정상적으로 실행하기 위한 전체 시스템 구성 기록(Spec Sheet)입니다.
이 문서는 추후 환경 재구축 시 100% 동일한 세팅을 복원할 수 있도록 작성되었습니다.

---

## 1. Windows Host GPU 환경

> 출처: `nvidia-smi`

| 항목 | 값 |
| :--- | :--- |
| NVIDIA-SMI 버전 | `555.59` |
| Driver Version | `556.13` |
| CUDA Version (Windows Runtime) | `12.5` |
| GPU 접근 | WSL2 GPU 가속 정상 동작 |

## 2. WSL2 / Ubuntu 환경

| 항목 | 값 |
| :--- | :--- |
| WSL 버전 | WSL2 |
| 배포판 | Ubuntu 22.04.5 LTS |
| 사용자 | `stone` |
| 프로젝트 env | `conda env pajusi` |

## 3. CUDA Toolkit (WSL Ubuntu 내부)

> 출처: `nvcc --version`
```
Cuda compilation tools, release 11.5, V11.5.119
Build cuda_11.5.r11.5/compiler.30672275_0
```

| 항목 | 값 |
| :--- | :--- |
| CUDA Toolkit Version | `11.5` |
| 빌드 버전 | `V11.5.119` |
| 비고 | Windows CUDA(12.5)와 달라도 WSL 구조상 문제 없음 |

## 4. cuDNN 설치 상태

> 출처: `dpkg -l | grep cudnn`
```
cudnn-local-repo-ubuntu2204-9.15.1
cudnn9-cuda-12                     9.15.1.9-1
cudnn9-cuda-12-9                   9.15.1.9-1
libcudnn9-cuda-12                  9.15.1.9-1
libcudnn9-dev-cuda-12              9.15.1.9-1
libcudnn9-headers-cuda-12          9.15.1.9-1
libcudnn9-static-cuda-12           9.15.1.9-1
```

| 항목 | 값 |
| :--- | :--- |
| cuDNN Version | `9.15.1.9` |
| 설치 대상 CUDA | CUDA 12.x |
| 구성요소 | runtime, dev, headers, static 모두 포함 |
| `LD_LIBRARY_PATH` | 기본값 (추가 설정 없음) |

## 5. Python / PyTorch / Whisper 환경

### Conda 환경
| 항목 | 값 |
| :--- | :--- |
| env | `pajusi` |
| Python | `3.10` |

### PyTorch 설정
> PyTorch CUDA build는 아래와 같이 구성되었습니다.

| 항목 | 값 |
| :--- | :--- |
| `torch` version | `2.5.x` (cu118 빌드) |
| `torch.cuda.is_available()` | `True` |
| `torch.version.cuda` | `11.8` |

### Faster-Whisper / CTranslate2
| 항목 | 값 |
| :--- | :--- |
| Whisper Backend | `faster-whisper` |
| 모델 | `Systran/faster-whisper-medium` |
| device | `cuda` |
| compute_type | `float16` |
| CTranslate2 | 정상 작동 |

## 6. STT 성능 측정 결과
> 테스트: 약 4.6초 길이 음성

| 항목 | 결과 |
| :--- | :--- |
| 모델 로드 | `3.81` 초 |
| STT 변환 | `0.16` 초 |
| GPU 동작 | 정상 |
| 출력 텍스트 | 파주시 고령층 대상포진 안내 문구 정확히 추출 |

## 7. 환경 백업 (필수)

> 아래 명령어를 통해 환경 구성 파일을 생성했습니다.
```bash
conda activate pajusi
pip freeze > requirements.txt
conda env export > env_pajusi_gpu.yml
```
**생성된 파일**
- `requirements.txt`
- `env_pajusi_gpu.yml`

> **참고**: 환경이 손상될 경우, 이 두 파일로 100% 복원 가능합니다.

## 8. 복원 방법 (Backup → Restore)

### 1) Conda 환경 복원
```bash
conda env create -f env_pajusi_gpu.yml
```

### 2) Pip 패키지 복원
```bash
pip install -r requirements.txt
```

## 9. 전체 아키텍처 요약

```
Windows NVIDIA Driver 556.13
        │
        ▼
WSL2 → GPU Passthrough (CUDA 12.5 Runtime)
        │
        ▼
Ubuntu 22.04 (WSL2)
        │
        ├─ CUDA Toolkit 11.5
        ├─ cuDNN 9.15.1 (CUDA 12.x용)
        │
        ▼
conda env pajusi
        ├─ Python 3.10
        ├─ torch 2.5.x (cu118)
        ├─ faster-whisper + CTranslate2
        └─ Systran/faster-whisper-medium (GPU float16)
```