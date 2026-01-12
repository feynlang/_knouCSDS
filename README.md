<!-- ------------------------------ -->
# Chapter 
## .
### 한줄 요약
### 키워드 흐름
### 막혔던 부분/질문

# ~~Chapter N.(legacy)~~
### Outline
### Flow Diagram
<!-- *** -->
## Main Concepts
### (minor heading)
#### (section)
<!-- *** -->
## Application
### example review
### confusing point
### extension point (implement)
## Reference (bookmark)


- "이 시스템 아키텍처가 왜 효율적인가?", "멀티 스레드 환경에서 데이터 정합성을 어떻게 지킬 것인가?" 같은 설계(Design) 능력과 문제 해결 능력은 Spring 공부를 통해 길러지며, 이는 에이전트가 함부로 대체하지 못합니다.
---
# Template Guideline.
>first, 간단메모 (디테일 빼고 구조(목차)만 빠르게 쓰기)\
>second, 강의 후 정제 (내 말로 재서술--구조에 채워넣기, 개념 네트워크 구성)

[A단계] 강의듣기, 간단메모(핵심만.밑줄 대용) (Raw Notes)
→ 목적: 구조 복원 + 강의 정보 확보
⚠ 이 단계에서는 해석 금지, “강의 복원”만 한다.

[B단계] 강의 후 정제 (20–30분)
→ 아래 템플릿을 따라 내용을 재배치
→ 목적: 개념 네트워크 + 능력 어필 요소 추가

[C단계] 분량 조절
A단계: 있는 그대로. 멈추고 해석하기 금지.
B단계 정제: 꼭 필요한 부분만 6개 항목으로 압축
포트폴리오에는 B단계만 업로드 (A단계는 로컬 보관)

---
## 그렇다면 “포트폴리오로 가치 있게” 보이려면?
포트폴리오로 쓰기 위한 노트의 조건은 명확하다:

### 조건 A. 원천 자료의 단순 복사가 아니라 재구성이어야 한다
예시:
- 여러 강의/책을 비교하며 차이점 정리
- 개념 간 연결(그래프 구조)
- 실수하기 쉬운 포인트 명시
- 문제 풀이 전략 정리
이렇게 되면 그 글이 당신만의 산출물이 된다.

### 조건 B. 내 사고 과정이 보이는 노트여야 한다
예시:
“처음엔 이 부분이 헷갈렸다 → 내가 만든 예제 → 해결 과정”
“수업 설명은 이랬지만 실제 코드는 이렇게 된다”
“책과 교수님 설명의 차이가 여기에 있다”
“이 개념은 ML 모델 구현의 이 파트와 연결된다”
이런 기록은 그 어떤 강의노트보다 강력한 포트폴리오다.

### 조건 C. 구현/코드/실험/연결 사례를 포함해야 한다
글과 함께 다음이 있으면 압도적으로 설득력 있다:
- DFA/NFA 변환 코드
- topological sort 시각화
- gradient descent toy example
- KNN 구현 시 느낀 bias-variance 정리
- SQL injection 방어 구현 실험
- CS 프로젝트에서 해당 개념이 어떻게 쓰였는지

이렇게 “이해→적용→반성”의 cycle이 들어가면
이건 더 이상 필기가 아니라 skill evidence다.

### 결론:
“강의노트를 블로그처럼 목차 + 세부내용으로 쓰는 것”
→ 학습 기록으로는 충분히 의미 있음
→ 하지만 포트폴리오 주력 콘텐츠는 되기 어렵다

“내 기준으로 재구성 + 예제/실험/코드 연결 + 의문·해결 과정 포함”
→ 이건 확실하게 가치 있는 포트폴리오가 됨

---
# 예시
>- 주제: 제어장치(하드와이어드 vs 마이크로프로그램 제어)
>- 범위: 강의 6·7강을 개념 중심으로 통합한 버전
>- 형식 깔끔, 정보량은 연구·실무·학습 재복습까지 충족하도록 설계(포트폴리오용)

# CH4. 제어장치(Control Unit)

## 1. 개요 요약 (5줄)
제어장치는 CPU 내부에서 **명령어를 해석하고 필요한 제어신호를 생성하는 핵심 모듈**이다.
데이터패스의 모든 동작(ALU 연산, 레지스터 읽기/쓰기, 메모리 접근 등)은 제어신호 조합에 의해 결정된다.
제어 방식은 크게 **하드와이어드(논리회로 기반)**와 **마이크로프로그램(제어메모리 기반)**으로 나뉜다.
하드와이어드는 빠르지만 유연성이 떨어지고, 마이크로프로그램은 느리지만 확장·수정이 용이하다.
ISA·CPU 성능·개발 비용·유연성 요구에 따라 적합한 제어 방식이 달라진다.
---

# **2. 핵심 흐름(Flow Diagram)**
```
          Instruction Register
                   │
                   ▼
            Opcode / Function
                   │
     ┌─────────────┴─────────────┐
     │                           │
 Hardwired                      Microprogrammed
 Control Logic                  Control Memory (ROM)
 (PLA, FSM, Combinational)      + Microsequencer
     │                           │
     ▼                           ▼
     Control Signal Generator (ALUop, RegWrite, MemRead, PCsrc, ...)
                   │
                   ▼
             Data Path 동작
```
---
# 3. 주요 개념 정리
## 3.1 제어신호(Control Signals) 개념
CPU 내부의 모든 구성요소는 제어신호에 의해 동작이 결정된다.
예:
* `RegDst` : 목적 레지스터 선택
* `ALUop` : ALU 연산 종류 지정
* `MemWrite/Read` : 메모리 접근
* `PCsrc` : 분기 여부 결정
제어장치의 핵심 목적은 **현재 명령어와 상태에 따라 이 신호들을 올바르게 생성하는 것**이다.

---
## 3.2 하드와이어드 제어 (Hardwired Control)
명령어 비트 패턴을 입력으로 하는 **논리회로(조합논리 + 순차 FSM)로 직접 제어신호를 생성**하는 방식.
### 특징
* 빠름(고정된 논리 경로)
* 소비전력 낮음
* 제어신호 변화가 cycle에 정확히 동기화됨
* 설계 변경 어려움 → **ISA 확장/변경 비용 큼**
### 구성 요소
* Opcode decoder
* Control signal generator
* FSM (pipeline stage 관리)
* PLA(Programmable Logic Array) 기반 제어도 사용됨
### 예시 (단일 사이클 MIPS 컨트롤)
```
Opcode bits ──► Decoder ──► Control Line Vector
                  │
                  ├─ RegDst = 1
                  ├─ ALUop = 10
                  ├─ MemRead = 0
                  └─ PCsrc = (Branch && Zero)
```
### 장점/단점
| 항목     | 하드와이어드              |
| ------ | ------------------- |
| 속도     | 매우 빠름               |
| 구조     | 단순(고정 회로)           |
| 수정     | 거의 불가능              |
| ISA 확장 | 어렵다                 |
| 복잡도    | 명령 많아질수록 급증(FSM 폭발) |

---
## 3.3 마이크로프로그램 제어 (Microprogrammed Control)
제어신호를 메모리(ROM)에 저장된 **마이크로명령어(microinstruction)**로 생성하는 방식.
### 핵심 개념
* 하나의 ISA 명령어 → **여러 개의 마이크로명령어 시퀀스**
* 마이크로명령어는 제어신호 비트를 포함
* 마이크로프로그램은 ROM·PLA에 저장
* 마이크로시퀀서(microsequencer)가 다음 마이크로명령을 결정
### 마이크로명령어 포맷(예시)
```
| NextAddr | Cond | ALUop | RegWrite | MemRead | MemWrite | PCsrc | ...
```
### 동작 흐름
```
명령어 Opcode ─► Microprogram ROM entry ─► Microinstructions sequence
                                          └► Data Path 제어
```
### 장점/단점
| 항목     | 마이크로프로그램             |
| ------ | -------------------- |
| 속도     | 하드와이어드보다 느림          |
| 구조     | 복잡 ↓ (제어의 구조화)       |
| 수정/확장  | 매우 쉬움                |
| ISA 확장 | 용이, CISC 구조와 매우 잘 맞음 |
| 유지보수   | 쉬움                   |

---
# **4. 내가 만든 예제 (Mini Microprogram Example)**
ISA 명령어:
```
ADD R1, R2, R3
```
### Microprogram(예시)
1. `FETCH`
   ```
   IR ← Mem[PC]; PC ← PC+4
   ```
2. `DECODE`
   ```
   A ← Reg[R2], B ← Reg[R3]
   ```
3. `EXECUTE-ALU`
   ```
   ALUout ← A + B
   ```
4. `WRITE-BACK`
   ```
   Reg[R1] ← ALUout
   ```
마이크로명령 1개가 실제로는 20~40비트 제어신호 집합을 포함할 수 있음.
이런 방식은 **CISC(x86) ISA처럼 복잡한 명령어를 단순한 내부 연산으로 감싸기 위한 구조**와 매우 잘 맞는다.

---
# **5. 오개념 방지 포인트 (학습하며 헷갈렸던 부분)**
1. **마이크로프로그램 = Firmware** 가 아니다
* firmware: 저장장치에 탑재된 프로그램
* microprogram: CPU 내부 제어ROM에 저장된 제어 시퀀스
  → 개념적으로 다르다.

2. 하드와이어드 제어도 **순차논리(FSM)** 를 포함한다
* 단순 조합 논리만 있는 것이 아니다.

3. CISC = 마이크로프로그램, RISC = 하드와이어드
* 일반적으로 맞지만 완전한 동일공식은 아니다.
* 현대 CPU는 **둘을 혼합**하기도 한다.

4. 마이크로프로그램 방식이 “느리다”
* 절대 속도 차이가 수십 배 단위는 아니다.
* 주로 cycle 증가 + 제어 ROM 접근 비용 때문에 느림.

---
# **6. 프로젝트/구현 연결 메모 (내 적용 사례)**
### (1) KNN/ML 프로젝트와 연관
캐시 locality 측면에서 데이터 접근 패턴을 최적화할 때
“메모리 read/write를 제어하는 제어신호 논리” 이해가 직접 도움됨.
### (2) 컴파일러 강의와 연관
명령어 디코딩 단계에서 control unit의 역할을 기반으로
“중간코드 → 기계어 매핑”을 어떻게 설계해야 하는지 이해 가능.
### (3) 단순 CPU 설계 toy project
Verilog로 5-stage pipeline 만들 때
`RegWrite`, `ALUop`, `PCsrc` 등 control signal의 생성 시점을 직접 설계해야 하므로,
하드와이어드 FSM 방식의 구조를 그대로 구현하게 됨.
### (4) RISC-V mini core reading
RISC-V 구현을 읽을 때 제어장치 구조를 이해하면
`control.v`, `decode.v` 파일의 구성을 빠르게 파악할 수 있음.

---
# **7. 강의 매핑 (기록용)**
* 이 Chapter는 **6강(제어장치 I) + 7강(제어장치 II)** 내용을 기반으로 하되
  강의 구성 순서가 아닌 **개념 중심 구조**로 재편성함.

---
# **8. ToC 링크(포트폴리오 업로드 시 권장 구조)**
프로젝트 repo에서 다음 식으로 목차 구성 가능:
```
computer-architecture/
 ┣ CH1-digital-logic.md
 ┣ CH2-ISA.md
 ┣ CH3-datapath.md
 ┣ CH4-control-unit.md   ← (본 문서)
 ┣ CH5-CPU-internals.md
 ┣ CH6-memory-hierarchy.md
 ┗ CH7-IO-parallelism.md
```