# part 1. 문제해결시작
# part 2. 알고리즘 분석


# part 3. 알고리즘 설계 패러다임
<!-- ------------------------------ -->
# Chapter 6. 무식하게 풀기
## Outline
## Flow Diagram
<!-- ************ -->
## Main Concepts
- 무식하게 푼다=brute-force
    - 컴퓨터의 빠른 계산능력을 이용->가능한 경우의 수를 일일히 나열하며 답찾기
    - 알고리즘: 완전 탐색(exhaustive search)

### 재귀호출과 완전탐색
- 컴퓨터가 수행하는 작업들: 대개 작은조각으로 쪼개기 가능
- 조각 범위가 작을수록 형태들이 유사해지는 작업⬆
    - ex. for반복문(완전 동일한 작업)
- 재귀호출(recursion, recursive function)을 사용
    1. n개의 조각으로 쪼개고
    2. 이 조각 중 하나를 **떼내어** 자신이 해결 후
        - 무엇을 떼내는지에 따라 해결여부가 갈림.
    3. 나머지 조각들을 자기자신이 호출하여 해결
        - 더 이상 쪼개지지 않는 최소한의 작업에 도달 시 답을 반환하는 조건문 포함 (기저사례,base case)
        - 모든 가능한 입력이 항상 기저사례의 답을 이용해 계산될 수 있어야함.
    ```cpp
    //1~n까지의 합을 구하는 문제
    int recursiveSum(int n){
        if(n==1) return 1; //기저 사례
        return n+recursiveSum(n-1);
    }
    ```
- 코딩을 간편하게 해줄 수 있는 강력한 무기O

#### 예제1: 중첩 반복문 대체하기
[예제1번](./code-snippet/p6_e1.cpp)
#### 예제2: 보글게임
#### 문제1: 소풍
#### 문제2: 게임판 덮기
### 최적화 문제
#### 예제1: 여행하는 외판원
#### 문제3: 시계 맞추기
#### 완전 탐색 유형
<!-- ************ -->
## Application
### example review
### confusing point
### extension point
(implement)
## Reference(bookmark)


## Chapter 7. 분할정복
## Chapter 8. 동적계획법
## Chapter 9. 동적계획법 테크닉
## Chapter 10. 탐욕법
## Chapter 11. 조합탐색
## Chapter 12. 최적화문제->결정문제로 풀기


# part 4. 유명한 알고리즘들
<!-- ------------------------------ -->
## Chapter 13. 수치해석
#### Outline
#### Flow Diagram
<!-- ************ -->
### Main Concepts
#### (minor heading)
##### (section)
<!-- ************ -->
### Application
#### example review
#### confusing point
#### extension point
(implement)
### Reference
(bookmark)
# part 5. 기초 자료구조
# part 6. 트리
# part 7. 그래프