# 1. 프로젝트 목적
HTML 문서를 파싱하여 DOM Tree 생성과정을 직접 구현
브라우저의 문서->DOM 변환의 핵심 원리 이해와 체득

# 2. 구현 범위(MVP)
- 단순 HTML 문법 지원
- 중첩 구조 파싱
- DOM Tree 생성

# 3. 미구현 기능
- 브라우저 구현의 응용/기술적인 부분
    - 에러 복구(malformed HTML), 예외처리
    - style, script 태그 부분

# 4. 설계 구조
1. Lexer -> Token Stream
2. Parser -> node -> DOM Tree

# 5. 활용되는 CS 개념

# 6. 한계 및 확장