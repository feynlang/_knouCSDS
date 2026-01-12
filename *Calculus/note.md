- [증명]
    - (1) Idea: (어디서부터시작?,어떤 성질이용?등)
    - (2) Key Trick:
    - (3) Conclusion: (증명의논리적결과)

# Chapter 
## .
### 한줄 요약
### 키워드 흐름
### 막혔던 부분/질문

<br/>

# Chapter 1. Functions and Model
## 1.4 exponential function
### 한줄 요약
1. $f=a^x$일때 $x=n,0,-n,\frac{p}{q}$에 따라 어떻게 정의되는지(**a는 양수**)\
    - -> 유리수일때 q의 값이 odd/even이냐에 따라 양수x 조건이 작용되는 포인트
    - -> 실수로 확장하는 방법(유리수x의 샌드위치 극한)
    - => 그래프 개형으로 확인($a\gt1$, $0\lt a\lt 1$, $a=1$),(지수함수$2^x$ vs 다항함수$x^2$)
2. 지수함수 성질(연속함수o,지수법칙o, 양/음의 무한대 극한값)
---
3. 오일러수 $e:=\lim_{x\to 0}(1+x)^{\frac{1}{x}}$ (exists, 무리수이자 초월수)\
    - -> natural exponential function (f(x)=e^x)
    - => guess: $f'(0)=1$ 
        - $e$의 정의에 따라 $(1+h)^{1/h}~=e \Rightarrow e^h-1~=h$ for small $h$
        - or {$e^x$의 테일러 전개를 활용
### 키워드 흐름
### 막혔던 부분/질문
- [ref] Sec 3.1(a) 지수함수 - (1/2),(2/2)

### 1.5 inverse functions and Logarithm
### 한줄요약
1. one to one function (injective,단사함수)
    - $\Leftrightarrow$ 정의역의 원소 $x$가 다르면 함숫값$f(x)$도 다름 (역도 성립, 증명의 key)
2. inverse function:
    - one-to-one function($f$: domain A, **range B**)$\to$($f^{-1}$: **domain B**, range A)
        - => cancellation equation: 모든 B(A)의 원소 x에 대해 $f\circ f^{-1}$ / $f^{-1}\circ f$ $=x$
        - => 그래프 개형 상, y=x(line)에 대칭
---
3. $f^{-1}$도 연속 $\Leftarrow$ $f$: one-to-one함수 $\land$ 연속 
    - *(증명은 <해석학>에서 다룸)*
4. 역함수의 도함수
    - $f$가 one-to-one함수이고 미분가능하며 $f'(f^{-1}(a))\ne 0$이면 
    - $(f^{-1})'(a)=\frac{1}{f'(f^{-1}(a))}$
    - [증명]
        $\lim\limits_{x\to a}\frac{f^{-1}(x)-f^{-1}(a)}{x-a}\\
        =\lim\limits_{y\to b}\frac{y-b}{f(y)-f(b)}
        =\frac{1}{\lim\limits_{y\to b}\frac{f(y)-f(b)}{y-b}}
        =\frac{1}{f'(b)}
        =\frac{1}{f'(f^{-1}(a))}$
    - [증명에 필요한 것]
        - (1) $f^{-1}$의 극한값이 존재하는지, 즉 연속인지 $\to$ 위 1번 정리 이용
        - (2) $f^{-1}(a)=b, f^{-1}(x)=y\\ \Rightarrow a=f(b), x=f(y)$
5. Leibniz notation: $\frac{dy}{dx}=\frac{1}{\frac{dx}{dy}}$
    - $\because$ $y=f^{-1}(x)\Rightarrow x=f(y)$이기에
        - 역함수 미분 형태인 $(f^{-1})'(x)=\frac{1}{f'(f^{-1}(x))}$ 에서 
        - 좌변은 y를 x로 미분한 것, 우변은 x를 y(=$f^{-1}(x)$)로 미분한 것의 역수가 됨
6. implicit differentation:
    -  $f^{-1}$가 미분가능하다면, 음함수 미분법을 사용하여 유도 가능
    - $f(y)=x$를 x에 대해 미분 
        - -> $f'(y)\frac{dy}{dx}=1\Rightarrow \frac{dy}{dx}=\frac{1}{f'(y)}=\frac{1}{f'(f^{-1}(x))}$
---
7. 지수함수의 역함수
    - [정의]: $a>0$, $a\ne 1$ 이면, $f(x)=a^x$는 one-to-on function $\Rightarrow$ 역함수 존재 $f^{-1}=log_ax$ (logarithmic function with base $a$)
        - $f(f^{-1}(x))=f^{-1}(f(x))=x$, 
        - 따라서 $a^{log_ax}=x$ for all $x\in\R$, $log_a(a^x)=x$ for all $x>0$임.
    - log함수 성질: if $x,y>0$, then
        - $log_a(xy)=log_ax+log_ay$
        - $log_a(\frac{x}{y})=log_ax-log_ay$
        - $log_a(x^r)=rlog_ax$ for any $r\in\R$
        - 극한: ($a>1$), $x\to\infty$일때 $\infty$, $x\to0^+$일때 $-\infty$
8. natural logarithms(base $e$): $\ln x:=log_ex$
    - => $\ln(e^x)=x$ for all $x\in \R$,
    - => $e^{\ln x}=x$ for all $x>0$
    - 밑 변환 공식(change of base formula): 양수a($\ne 1$)이면 $\log_ax=\frac{\ln x}{\ln a}$ $\to \log_ax=\frac{1}{\log_xa}$ for $x\ne 1$
        - [증명]: 역함수인 지수함수 합성 후 밑이 e인 $\ln$을 양변에 취하여 유도
            - $y=\log_ax \\
            \to a^y=log_ax \\
            \to \ln(a^y)=\ln(x) \\
            \to y\ln a=\ln x$
### 키워드 흐름
### 막혔던 부분/질문
- [!]도메인과 range 표현시 주의
- [Q]2^x=x^2을 대수적으로 푸는 방법?
- [ref] Sec 3.2(a) Inverse Functions and Logarithms - (1/3),(2/3),(3/3)

<br/>

# Chapter 2. Limits and Derivatives
## 2.1 Tangent(Velocity Problem)
## 2.2 Limit of a Function
## 2.3 Calculating Limits (limit laws)
## 2.4 Continuity
## 2.5 Limits at Infinity
## 2.6 Derivative & Rates of change
## 2.7 Derivative as a Function

<br/>

# Chapter 3. Limits and Derivatives
## 3.1 Derivatives 1(polynominals, exponential)
## 3.2 Product & Quotient Rules
## 3.3 Chain Rule
## 3.4 Implicit Differentiation
## 3.5 Derivatives 2(Logarithmic, Inverse Trigonometric)
1. 로그함수의 미분: $f(x)=\log_ax$
    - $f'(x)=\frac{1}{x}\log_ae=\frac{1}{x\ln a}$ and $\frac{d}{dx}(\ln x)=\frac{1}{x}$
    - [증명] key: 미분계수 정의에 따른 식조작 시 오일러수의 정의(극한)을 활용
    - -> +chain rule:
        - $\frac{d}{dx}[\log_ag(x)]=\frac{g'(x)}{g(x)\ln a}$
        - $\frac{d}{dx}[\ln g(x)]=\frac{g'(x)}{g(x)}$

2. $\ln |x|$는 미분가능함(!)
    - $\frac{d}{dx}\ln |x|=\frac{1}{x}$ (for $x\ne 0$)
    - [증명]
        - $x>0,\ f(x)=\ln x$ => $f'(x)=\frac{1}{x}$
        - $x<0,\ f(x)=\ln(-x)$ => $f'(x)=\frac{(-x)'}{-x}=\frac{1}{x}$
---
3. 로그미분법: 몫의 미분 대신 로그성질(+,-로 치환되는)을 이용하여 수월함
    - (1) 양변에 자연로그를 취한다
    - (2) x에 대해 미분한다
    - (3) y'에 대해 푼다(y는 처음 주어진 x에 대한 식으로 치환)
    - -> power rule 증명 가능 ($f(x)=x^n$, then, $f'(x)=nx^{n-1}$)
        - [증명]: 모든 실수 n에 대한 증명
            - $|y|=|x|^n\Rightarrow \ln|y|=n\ln|x|$ for  $x\ne0$ $\Rightarrow\frac{y'}{y}=\frac{n}{x}\Rightarrow y'=y\cdot\frac{n}{x}=nx^{n-1}$
4. 지수함수 미분: 
    - $f(x)=a^x$($a>0$)는 미분가능
    - $\frac{d}{dx}(a^x)=a^xln a$ ($\frac{d}{dx}(e^x)=e^x$)
    - [증명]
        1. $y=\log_ax$는 one-to-one, 미분가능, $y'=\frac{1}{x\ln a}$ 는 nonzero
        2. 따라서 inverse function도 미분가능
        3. $y=a^x \to \log_ay=x$ ->(x에 대해 미분) $\frac{y'}{y\ln a}=1\Rightarrow y'=y\ln a=a^x\ln a$
    - -> example: $f(x)^{g(x)}$ 미분
        - sol1: exponential
            - $y=f(x)^{g(x)}=e^{\ln(f(x)^{g(x)})}=e^{g(x)\ln(f(x))}$,
            - $y'=e^{g(x)\ln(f(x))}\cdot(g(x)\ln(f(x)))'=f(x)^{g(x)}\cdot(g(x)\ln(f(x)))'$
        - sol2: logarithmic diff
            - $\ln y=g(x)\ln(f(x))$
            - $\frac{y'}{y}=(g(x)\ln(f(x)))'\\ \Rightarrow y'=y\cdot(g(x)\ln(f(x)))'\\=f(x)^{g(x)}\cdot(g(x)\ln(f(x)))$
---
1. 
---
1. 

### 키워드 흐름
### 막혔던 부분 
- [ref] Sec 3.3(a) Derivatives of Logarithmic and Exponential - (1/2),(2/2)
- [ref] Sec 3.5(a) Inverse Trigonometric Functions - (1/2),(2/2)

## 3.8 Exponential Growth & Decay
1. natural growth/decay: 미분방정식 형태
    - (증가/감소량이 그 순간의 사이즈에 비례 proportional to their size)
    - $y(t)$: 시간 t에 따른 수량 y의 함수 => $\frac{dy}{dt}=ky$(k는 상수)
    - 솔루션: only, $y(t)=y(0)e^{kt}$
        - $y(t)=Ce^{kt}$ => $y'(t)=Cke^{kt}=ky(t)$
        - $y(0)=C\cdot 1$
---
2. Population Growth
3. Newton's Law of Cooling
### 키워드 흐름
### 막혔던 부분/질문
- [ref] Sec 3.4(a) Exponential Growth and Decay - (1/2)

## 3.9 Related Rates
## 3.10 Linear Approximations
## 3.11 Hyperbolic Functions
Sec 3.6(a) Hyperbolic Functions - (1/2),(2/2)


<br/>

# Chapter 4. Applications Of Differentiation
## 4.1 Maximum/Minimum Values
## 4.2 The Mean Value Theorem
## 4.3 Derivatives $\to$ Shape of Graph
## 4.4 Indeterminate Forms (L'Hospital's Rule)
Sec 3.7(b) Indeterminate Forms and L'Hospital's Rule -(1/2),(2/2) 
## 4.5 Curve sketching
## 4.6 Graphing(calculus/tech)
## 4.7 Optimization problems
## 4.8 Newton's Method
## 4.9 Antiderivatives