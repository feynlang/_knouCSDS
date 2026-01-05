# Chapter 1. 웹 프로그래밍 개요
> 웹 개요/웹페이지 구성/HTML/웹 프로그래밍 과정
### Outline
웹고 구체적인 웹 페이지의 구성과 구조에 대해~~
### Flow Diagram
<!-- *** -->
## Main Concepts
### 웹 개요 -(1/3)
1. 용어
- 웹 문서vs웹 페이지: html 포맷 vs 브라우저 창에 표시(action O)
    - 웹 사이트: 웹페이지 묶음단위(semantic)
- 웹 서버-클라이언트 구조(제공-사용)
- Internet != World Wide Web
    - derived by: ARPANET vs Tim Berners-Lee
    - 컴퓨터 네트워크 시스템 vs 하이퍼링크-GUI-HTTP 기반 서비스 
2. http 과정 (\*:HTTP 세션)\
(1) 브라우저 - url 입력 (url: 프로토콜//서버주소:포트번호/폴더명 및 파일명 으로 구성)\
(2) DNS서버 - IP주소변환\
(3) 브라우저 - HTML 페이지요청*\
(4) 웹서버 - 내부 페이지 요청 및 전송*\
(5) 웹서버 - HTML 파일전송\
(6) 브라우저 - HTML 요소 분석 및 변환
### 웹 페이지 구성 -(2/3)
- HTML: 구조,내용 **정의**
- CSS: 출력모양,디자인 **정의**
- JavaScript: 동작,상호작용 **처리**
### 웹 프로그래밍 과정 -(3/3)
1. 웹사이트/웹페이지 만들기=>Front-End 프로그래밍
- 페이지제작: html,css,js
- 프레임워크: react,angular,vue.js
2. html 문서: 요소(시작태그,내용,종료태그)들의 조합
- 시작태그: `<요소명 속성명="속성값"...>`
- 종료태그: `</요소명>`
- notice
    - 파일명.html (or .htm)
    - 화이트스페이스: 두칸 이상의 연속 공백=공백 하나로 처리
    - 요소명,속성명에 대소문자 구분X
    - 요소 간의 포함관계O
- 구조
    - `<!DOCTYPE html>`: 반드시 첫줄, 문서형식 지시어
    - `<html> ... </html>`: 웹페이지의 루트요소(문서의 시작과 끝)
    - `<head> ... </head>`: 웹문서의 메타요소(화면표시X)
        - title요소(제목): 브라우저 상단에 표시되는 문서제목
        - meta요소(메타데이터): `charset="문자인코딩방식"`, `name="description|keywords|author"`--`content="문서설명|키워드|문서작성자"`
        - script요소(자바스크립트): js코드 작성(or외부코드링크 로드)
        - style요소(css): 웹문서 전체에 적용할 스타일 정의
        - link(css): *.css 파일을 가져와 문서에 적용
        - base(기본url경로): href(body 요소 내 상대경로->절대경로 변환), target(출력 타깃 윈도우)
    - `<body> ... </body>`: 실제 출력되는 본문의 모든 내용(자바스크립트 코드o)
    - `<!-- -->`: 주석문(주석 안에 '<','>','--'사용불가)
<!-- *** -->
## Application
### example review
### confusing point
### extension point
## Reference(bookmark)
- knou HTML5 웹 프로그래밍 1강
- HTML5 웹 프로그래밍(이관용 저) ch1

<!-- ------------------------------ -->
# Chapter 2. HTML5
> 텍스트표현/리스트표현/이미지표현/하이퍼링크표현\
> /테이블표현/미디어표현/시맨틱요소/웹폼
### Outline
### Flow Diagram
<!-- *** -->
## Main Concepts
### 텍스트 표현 -(1/8)
1. br요소: 줄바꿈 (종료태그X)
2. p요소: 단락 나누기 (단락 구분 빈줄 자동추가)
3. hr요소: 수평선 (자동줄바꿈, 종료태그X)
4. h*n*요소: 단락의 제목 지정 (h1~h6, 볼드체 & 자동줄바꿈)
5. pre요소: 텍스트 표시 (문자가 입력된 형태 유지 출력)
6. blockquote요소: 인용문 작성 (상하 빈줄로 구분 & 들여쓰기)
    - cite속성: 인용 부분 url 지정
7. 공백 및 특수문자 (연속공백, 예약어문자, 키보드입력x문자)
    - &lt(<), &gt(>), &amp(&), &quot("), &nbsp(공백), &rarr(->), &copy(©)
    - 특수이름 or 10진수 코드 or 16진수 코드
8. 글자 스타일 지정 요소
- 물리
    - b요소(볼드/강조의미x주목o), i요소(이탤릭체/다른 어조), s요소(취소선/관련없는 내용), u요소(밑줄/불충분한 내용)
    - sup요소(위첨자), sub요소(아래첨자), small요소(작은크기/부수적인 해설)
- 논리
    - cite요소(작품제목), code요소(컴퓨터코드), samp요소(프로그램샘플)
    - var요소(변수표시), dfn요소(용어정의), em요소(강조표시)
    - strong요소(중요한 내용), abbr요소(축약형), address요소(실제 우편물주소, 상하자동줄바꿈)
    - kbd요소(키보드입력내용), q요소(짧은 인용구,`" "`), mark요소(시각/의미적 특정문구/단어 강조, `형광표시`)
9. 콘텐츠 컨테이너 요소(블록 vs 인라인)
- 블록 요소: 문서구조요소. 하나의 줄 공간 차지, 자동줄바꿈o
    - div요소: html요소를 묶어 구분된 논리적단위 영역 구성(->css 지정 or 레이아웃 구성)
- 인라인 요소: 텍스트요소, 입력내용만 공간 차지, 자동줄바꿈x
    - span요소: 텍스트를 위한 컨테이너 요소(->일부만 css 지정 or js 조작)
### 리스트 표현 -(2/8)
1. 순서 없는 리스트(Unordered List)
- ul요소: 각 항목 앞에 글머리 기호를 붙여 표시
    - 하위요소: li요소(들여쓰기, 자동줄바꿈, 종료태그없이사용o)
    - 글머리 기호 변경: css(list-style-type)
2. 순서 있는 리스트(Ordered List)
- ol요소: 각 항목 앞에 순서 기호를 붙여 표시
    - 하위요소: li요소(속성 `value="숫자"`, 항목번호를 중간에 바꿈)
    - 속성
        - `type="값"`: 순서 기호 유형->숫자(기본값), 영문자, 로마자 등
        - `start="숫자"`: 항목 시작 번호
        - `reversed`: 항목 번호 역순
3. 서술 리스트(Description List)
    - 하위요소: dt요소(용어/이름 나열), dd요소(이에 대한 설명 제공, 자동 줄바꿈 & 들여쓰기)
### 이미지 표현 -(3/8)
1. img요소: 종료태그x
- 속성
    - `src="파일url"`: (필수)이미지 파일 경로/이름
    - `width="숫자"`, `height="숫자"`: 폭과 높이
        - 픽셀 or %, 폭/넓이 중 하나만 지정->나머지는 원본크기에 비례
    - `alt="텍스트"`: 대체 표시 텍스트(이미지 표시가 안되는 경우 대비)
    - `usemap="#맵이름"`: 클라이언트 측 이미지맵
### 하이퍼링크 표현 -(4/8)
1. a요소: 하이퍼링크 삽입
- 속성
    - `href="URL"`: 링크로 이동하려는 곳의 경로/주소
        - href="[페이지경로]#이동할 특정위치 이름"
        - 다양한 형태 링크 지정O (`"mailto:메일주소"`, `"javascript:자바스크립트코드"`, `"song.mp3"` 등)
    - `id="이동할 특정위치 이름"`: 이동할 특정위치
    - `target="창이름"`: 링크된 내용이 출력될 창
        - "창이름", "_blank"(새로운 창), "_self"(기본값,현재창)
    - `download`: href에 지정된 파일을 다운로드/저장
2. 이미지맵(map요소)
- img요소의 속성 `<img usemap="#맵이름...>"`
- 하나의 이미지를 여러 영역으로 분할 후 링크 지정(속성: `name="맵이름"`)
- 하위요소: area요소
    - 속성: `shape="영역모양"`, `coords="영역구성좌표값들`, `href="URL"`, \
    `target="[출력]창이름"`, `download`, `alt="대체텍스트"`
3. iframe요소: 현재 웹페이지 안에서 다른 웹페이지창을 삽입
- 속성
    - `name="이름"`: 창이름(a태그의 target속성값에 이용)
    - `src="URL"`: 표시될 문서url(srcdoc이 있으면 무시됨)
    - `srcdoc="HTML코드"`: 표시될 html콘텐츠
    - `width="폭"`:픽셀값
    - `height="높이"`:픽셀값
### 테이블 표현 -(5/8)
1. table요소: 테이블 정의
    - m행 n열: m개의 tr요소마다 n개의 td요소
    - 속성: `border="숫자"`(테두리 지정--0은 사용x, 테두리 스타일은 css사용)

2. (table의 하위요소) caption 요소(테이블 전체에 대한 제목)
    - 레이아웃 영향x
    - table요소 시작태그 바로 뒤에 위치(그 외 위치에선 무시됨)
    - 표시: 테이블 상단, 볼드 & 가운데정렬
        - css(caption-side, text-align)으로 변경가능
3. (table의 하위요소) colgroup요소(열 단위 그룹핑)
    - 하나 이상의 열을 그룹->스타일지정
    - 하위요소(자식요소): col요소
        - 지정된 열에 각각의 포매팅 정의
        - 종료태그X
        - 속성: `span="n"`(스타일이 적용되는 열의 수), `class="클래스이름"`(css지정 시)
4. (table의 하위요소) thead, tbody, tfoot요소(행 단위 그룹핑)
    - 헤더, 몸체, 푸터 부분 구분(순서대로 사용)
    - 레이아웃 영향x
5. (table의 하위요소) tr요소(table row, 행)
    - td(or th)요소를 필수 포함

6. {tr의 하위요소} td요소(table data, 표준 셀)
    - (텍스트,이미지,테이블 등) 데이터 표시, tr요소 내에서만 사용, 왼쪽 정렬
    - 속성: `rowspan="n"`, `colspan"n"`(현재 셀 위치에서 인접한 n개 행/열을 병합)
7. {tr의 하위요소} th요소(table header, 헤더 셀)
    - 행/열의 제목 표시, 볼드 & 가운데 정렬, td요소 위치에서 선택적사용
    - 속성: `rowspan="n"`, `colspan"n"`
### 미디어 표현 -(6/8)
1. audio요소: 웹페이지에서 오디오 재생
- 속성
    - `src="URL"`: 오디오파일
        - 구간지정: `src="파일명#t=[시작시간-시:분:초][,종료시간-시:분:초]"`
    - `autoplay`: 로딩시 자동재생여부
    - `controls`: 오디오제어버튼 표시여부
    - `loop`: 반복재생여부
    - `muted`: 음소거여부
    - `preload="값"`: 로딩방식 지정(`none`:미리 로드x, `auto`:기본값,로딩 시 전체 자동로드, `metadata`: 오디오 메타파일만 로드)(`autoplay` 속성 지정 시 무시)
2. source요소: audio, video의 하위요소(src속성을 대체)
- 서로 다른 형식의 파일을 동시 지정
- 속성
    - `src="URL"`: 미디어파일
        - 구간지정: `src="파일명#t=[시작시간-시:분:초][,종료시간-시:분:초]"`
    - `type="MIME형식"`: audio/mp3, audio/wav, audio/ogg
    - `media="미디어쿼리"`: 미디어쿼리 지정
3. video요소: 비디오 재생
- 속성
    - `src="URL"`: 비디오파일
        - 구간지정: `src="파일명#t=[시작시간-시:분:초][,종료시간-시:분:초]"`
    - `autoplay`,`controls`,`loop`,`muted`,`preload`
    - `width="픽셀"`,`height="픽셀"`: 콘텐츠 표시영역 크기
    - `poster="이미지파일URL"`: 설정x/로딩중 표시될 이미지
### 시멘틱 요소 -(7/8)
- 기존방식: div로 나누어 id부여하는 식
- 표현중심x, 의미중심 구조의 문서 작성 요소
1. 문서구조화
    1. header요소: 머리말(로고, 사이트이름, 소개정보)
        - 위치무관: 내용중간(section/article)에서 사용시 머리글/바닥글
        - 포함불가 요소: section, header, footer요소
    2. footer요소: 꼬리말(제작자 정보, 저작권 정보, 연락처)
    3. nav요소: (페이지 전체에 적용) 상단메뉴표시 링크모음
    4. aside요소: 주요(x) 관련(o)컨텐츠 표시
    5. section요소: 하나의 주제로 구성된 **내용영역**
        - 중첩 사용 가능(하나의 주제 속 다른 세부주제)
        - 1개의 헤딩 요소만(하나의 주제 안에 여러개->레벨에 따라 묵시적 분리or중첩 구분)
    6. article요소: 독립적인 분배/재사용 가능한 **개별콘텐츠** (section요소와 분리o, section/article요소 포함가능)
2. 블록요소
    1. main요소: body영역의 주요 콘텐츠 블록 지정
        - 유일한 부분(1번만 사용), 반복적으로 사용되는 콘텐츠 포함X
        - 전체 레이아웃에 영향X
        - 문서구조화 요소(1번항목, section제외)의 하위요소로 사용불가
    2. figure요소(~figcaption요소): 삽입 그림 블록화
        - 목적: 독립적콘텐츠->하나로 묶어 블록화 / 캡션포함->관계를 명확히 구조화
        - 위치: 주요콘텐츠와 독립적(삭제해도 흐름영향x)
        - (자식요소) **figcaption요소**: figure에 대한 캡션 표시(부모요소 바로 다음or맨마지막)
    3. details요소(~summary요소): 사용자 요구(핸들클릭)에 따라 세부정보(콘텐츠 종류 제한x)를 선택적 표시
        - 속성: `open`(세부정보가 보여지도록 지정)
        - (자식요소) **summary요소**: 세부정보에 대한 캡션 지정
            - 캡션클릭->세부정보 표시여부 선택 가능(=핸들클릭)
            - 미사용->"세부정보"(기본제목 표시)
3. 인라인요소
    1. time요소: 시간/날짜 표현에 의미부여
        - 속성: `datetime="년-월-일"`(시스템 인식 목적-정확한 형식 시간/날짜)
    2. meter요소: 일정 범위 안 측정값/분포비율 표시 (정적인 값)
        - 속성: `value`(현재측정값), `min`,`max`(최소값-디폴트0.0, 최대값-디폴트1.0), `low`,`high`(허용범위의 최소값/최대값), `optimum`(최적값)
    3. progress요소: 현재 작업진행률 (동적인 값)
        - 속성: `value`(현재상태값),`max`(작업량 최대값-디폴트1.0)
        - 파일복사or다운로드 등, 지속적 업데이트 필요(js와 연동 필수!)
    4. ruby요소(~rt요소): 루비 주석이 달릴 기본텍스트
        - 자식요소: rt요소(바로 앞의 1개 이상의 구문에 대한 루비주석)
    5. wbr요소: 줄바꿈 허용위치 지정(무조건 수행을 의미x 레이아웃 배치에 따라 결정o)
        - 긴 URL, 단어, 텍스트 표시 시 유용

### 웹 폼 -(8/8)
1. form요소
- 폼요소(유저로부터 다양한 데이터를 입력받음)들을 담는 컨테이너
    - 폼요소(form요소 내에서만 사용): input요소, textarea요소, select요소, datalist요소, output요소, fieldset요소, label요소
- 속성
    - `action="URL"`(데이터를 전달받아 처리할 서버측 응용프로그램 주소), `method="전송방식"`(get/post),
        - method="get": 기본값, 폼데이터가 URL에 포함(`URL?name=value&.."`), 최대 2048글자, 이전페이지 이동 기능 가능, 보안이 중요하지 않은 데이터 전송 
        - method="post": HTTP Request몸체 속 포함, 길이제약X, 이전페이지 이동 기능-데이터 재전송 경고메세지O, 보안이 필요한 데이터 전송(로그인,회원정보 등)
    - `name`(폼이름), `accept-charset`(문자인코딩방식), `enctype`(post방식때의 데이터인코딩방식),
    - `autocomplete`(자동완성기능 사용여부), `novalidate`(데이터유효성검사X), `target`(서버로부터 전달받은 처리결과를 보여줄 창)
2. (폼요소) input요소: 다양한 유형 데이터를 입력받음(타입 지정)
- 종료태그X
- 속성: 
    - `type="입력타입"`: text, password, radio, checkbox, hidden, file, image, submit, reset, button,\
    search, tel, url, email, number, range, color,\
    date, month, week, time, datetime-local(날짜+시간)
    - `속성="값"`: tyep속성 값에 따라 사용가능한 속성이 다름.
3. 데이터 입력(input요소)
    1. 일반 텍스트입력
        - `type="text"`: 한줄-짧은 텍스트 입력or출력용도(속성: `value="초기표시텍스트"`)
        - textarea요소: 여러줄 텍스트 입력or출력용도
            - 속성: `rows="영역높이"`,`cols="영역너비"`,`name="텍스트영역이름"`,`wrap="soft(default)|hard`(줄바꿈포함여부),`readonly`(읽기전용)\
            `autofocus`,`disabled`,`form`,`maxlength`,`placeholder`,`required`
    2. 특정 형식 텍스트입력 (text타입과 모양 및 속성 동일;입력대상을 명확하게 하는 효과)
        - 비밀번호 `type="password"` (입력문자가 다른기호로 대체되어 표시)
        - 검색어 `type="search"`
        - 전화번호 `type="tel"`
        - URL `type="url"` (자동 유효성검사 수행)
        - 이메일주소 `type="email"` (자동 유효성검사 수행)
    3. 숫자 입력 (특정 범위)
        - 이산: `type="number"`
        - 범주: `type="range"`
        - 속성: `min`,`max`(입력가능한 최소/최대값), `step`(유효간격), `value`(초기값or현재입력값)
4. 데이터 선택
    1. 항목선택(input요소)
        - 라디오버튼 `type="radio"`(같은그룹 중 1개만 선택가능)
        - 체크박스 `type="checkbox"`(동시선택가능, 기호성향파악)
        - 속성: `name="이름"`(같은그룹은 동일한이름), `value="텍스트"`(서버로 전달->name="value"), `checked`(초기값 선택항목지정)
    2. 파일선택(input요소)
        - `type="file"`: 자신의 시스템에 있는 파일 선택 버튼
         - 속성: `accept="파일유형"`(선택가능한 파일유형 지정), `multiple`(동시에 여러파일선택가능)
    3. 데이터리스트상의 선택
        - select요소: 드롭다운 리스트
            - 속성: `name="이름"`(드롭다운리스트 이름), `size="숫자"`(한번에 화면에 보이는 항목개수), `multiple`(여러항목을 한번에 선택가능),\
            `autofocus`, `disabled`, `form`, `required`
        - datalist요소: `<input type="text"...>`이 포커스를 받을때 이에 대한 옵션리스트가 드롭다운형태로 표시
            ```
            <input type="text" list="데이터리스트아이디">
            <datalist id="데이터리스트아이디">
                <option value="값" ...>
                ...
                <option value="값" ...>
            </datalist>
            ```
        - (select/datalist의 하위요소)option요소: 각 옵션 정의
            - 속성: `disabled`(해당옵션 비활성화), `selected`(페이지 로드시 미리 선택될 초기옵션), `value="텍스트"`(서버로 전달될 항목값)
        - optgroup요소: option요소 그룹핑
            ```
            ..(중략)
            <optgroup>
                <option value="값1">값1</option>
                <option value="값2">값2</option>
            </optgroup>
            ..(중략)
            ```
    4. 색상선택(input요소): `type="color"`
    5. 날짜 및 시간선택(input요소): `type="date|month|week|time|datetime-local"` (날짜-년,월,일 | 년,월 | 년,주 | 시간-시,분,초 | 날짜,시간)
        - 속성: `min`,`max`,`step`,`value`
    6. 버튼 생성
    - (input요소의)`type="버튼종류"` (속성: `value="버튼명"`--버튼 위에 표시되는 이름)
        - 제출버튼: `type="submit"`(form요소에서 지정한 action속성의 서버로 전송) (기본버튼명: 제출)
        - 리셋버튼: `type="reset"`(입력중인 모든 데이터를 지움) (기본버튼명: 초기화)
        - 이미지버튼: `type="image"`(이미지를 이용한 제출버튼)
            - 속성: `src`,`width`,`height`,`alt`
        - 일반버튼: `type="button"`(클릭을 통해 자바스크립트 코드수행 시)
    - button요소
        - `<button> type="button|submit|reset" name="이름" value="초기값"</button>`
        - 버튼 위 텍스트/이미지 표시가능(하위요소로 삽입)
5. 그외
    1. 출력: output요소
        - `<output name="이름" for="출력할 연산식 파라미터"></output>`
        - 특정 input요소에 입력된 값이 변하면 form요소의 이벤트 발생, 자바스크립트 문법`oninput="연산코드"`으로 처리 후 이를 반영하여 화면 출력할때 사용됨.
    2. 폼요소-캡션 묶기: label요소
        - input요소에 대한 레이블 정의
        - 시각적 효과X, 마우스로 label요소 내의 텍스트(캡션) 클릭 시 포커스/선택이 가능해짐.
    3. 폼요소 그룹핑: fieldset요소~legend요소
        - fieldset요소
            - 사각형테두리 표시O(관련요소 포함)
            - 속성: `name`,`form`,`disabled``
        - legend요소(하위요소): 그룹핑되는 관련요소에 대한 캡션(사각형 테두리 선상에 표시O)
<!-- *** -->
## Application
### example review
### confusing point
### extension point(implement)
## Reference(bookmark)


<!-- ------------------------------ -->
# Chapter 3. CSS3
>css개요/선택자/색상/글꼴/텍스트/리스트\
/테이블/박스모델/테두리/배경/그라데이션\
/변형/전환/애니메이션/다단
### Outline
### Flow Diagram
<!-- *** -->
## Main Concepts
### CSS개요 -(1/15)
- Cascading Style Sheet, 웹문서 출력에 스타일 추가 매커니즘
- 형식: `선택자 { 선언블록(선언(속성:값); 선언; ...) }`
    - 선택자 selector: 스타일 적용대상 HTML요소
    - 속성 property: 선택자로 지정된 요소의 어떤 속성에 적용할것인지
    - 값 value: 해당속성에 대해 지정하는 스타일값
- 적용방법 (우선순위: style속성 > style요소 > 외부파일 > 브라우저디폴트)
    1. style요소 (임베디드 방법): head요소 내의 style요소
    2. style속성 (인라인 방법): 스타일을 지정할 요소의 속성
    3. 외부파일 (링크 방법): head요소 내의 link요소(`rel="stylesheet" href="css파일경로"`) or style요소 내 `@import url("css파일경로")`

### 선택자 -(2/15)
1. 기본 선택자
- 스타일을 적용하려는 HTML요소를 지정하기위한 패턴
- 종류
    1. 전체 선택자: `*{속성:값;}`(웹 문서의 모든 요소에 적용)
    2. 타입 선택자: `HTML요소이름{속성:값;}` (스타일을 지정하고자하는 요소 이름 직접 사용)
    3. 클래스 선택자
        - 같은 요소, 다른 스타일: 요소명 동일, 클래스명 다르게 지정=>`요소명.클래스명{속성:값;}`
        - 다른 요소, 같은 스타일: 요소명 생략, 클래스명 지정=>`.클래스명{속성:값;}`
        - 두 경우가 충돌 시->더 구체적인 지정한 선택자(`요소명.클래스명`)가 적용됨
    4. 아이디 선택자: `요소명#아이디명{속성:값;}`(#기호 + id속성)
        - id속성: 요소의 정체성(문서 내 중복X)
    5. 속성 선택자: 요소 속성값에 대한 표현과 일치하는지\
        `요소명[속성명 연산자 값] {속성:값;...}`
        - `요소명[속성명]`: 해당속성명 사용하는 모든 요소
        - `요소명[속성명="값"]`: 속성값이 일치하는 모든 요소
        - `요소명[속성명~="값"]`: 속성값에 값을 포함하는 모든 요소
        - `요소명[속성명|="값"]`: 값과 일치하거나 "값-"으로 시작하는 속성값을 가진 모든 요소
        - `요소명[속성명^="값"]`: 속성값이 값으로 시작하는 모든 요소
        - `요소명[속성명$="값"]`: 속성값이 값으로 끝나는 모든 요소
        - `요소명[속성명*="값"]`: 속성값이 값이라는 부분문자열을 포함하는 모든 요소
2. 가상 선택자: 어떤 상황이 발생한 경우에만 적용되는 선택자 지정
- 가상 클래스(`기본선택자:가상클래스`)
    1. 링크: `:link`(아직 방문X 링크 선택), `:visited`(방문한 적 있는 링크 선택)
    2. 사용자 동작: `:hover`(마우스가 올라간 링크 선택), `:active`(활성화-클릭-되는 링크 선택), `:focus`(포커스를 가진 요소 선택)
    3. UI 상태: `:enabled`(활성화된 상태의 요소 선택), `:disabled`(비활성화된 상태의 요소 선택), `:checked`(체크된 input요소-체크박스,라디오버튼-선택)
    4. 부정: `:not(X)`(X로 표현되지 않은 요소 선택)
    5. 트리 구조: `:root`, `:nth-child(n)`(동일한 부모요소 안에서- n번째 자식요소 선택), `:first-child`, `:last-child`,\
    `:nth-last-child(n)`(동일한 부모요소 안에서- 끝에서 n번째 자식요소 선택), `:nth-of-type(n)`(동일한 부모요소 안에서- 형제요소중에 n번째  요소 선택), `:first-of-type`(= :nth-of-type(1) ),\
    `:only-child`, `:only-of-type`, `:empty`
- 가상요소(`기본선택자::가상요소`)
    1. `::first-letter`: 특정 요소 첫번째줄 선택
    2. `::first-line`: 특정 요소 첫번째 글자 선택
    3. `::before`: 요소 앞 내용/스타일 추가(`{content:"추가할내용"; 속성:값; ...`)
    3. `::after`: 요소 뒤 내용/스타일 추가(`{content:"추가할내용"; 속성:값; ...`)
3. 결합자
- 두 개 이상 선택자 결합(문맥/요소 구조 기반)
- 종류
    - 자손(하위)결합자: 공백(`선택자1 선택자2`)
    - 자식결합자: >(`선택자1>선택자2`)
    - 인접형제결합자: +(`선택자1+선택자2`)
    - 일반형제결합자: ~(`선택자1~선택자2`)
    - 그룹결합자: ,(`선택자1,선택자2`)

### 색상 -(3/15)
1. 색상 지정 방식
    1. 색상명 사용(영어단어)
    2. RGB 색모델(3원색 성분의 합)
    - 16진수 사용: #RRGGBB (RR,GG,BB는 00~ff사이의 16진수)
    - RGB함수 사용: 
        - rgb(R,G,B): R,G,B는 0~256사이의 10진수
        - rgba(R,G,B,Alpha): alpha는 투명도(투명0.0 ~ 불투명1.0)
        - rgb(R%,G%,B%): R,G,B는 백분율 값
    3. HSL 색모델(색조Hue,채도Saturation,밝기Lightness)
    - HSL함수 사용
        - hsl(색조,채도,밝기): 색조=색상환의 각도(빨강0도 ~ 초록120도 ~ 파랑240도), 채도=(무채색0% ~ 순색100%), 밝기(검은색0% ~ 평균50% ~ 흰색100%)
        - hsla(색조,채도,밝기,Alpha): alpha는 투명도(투명0.0 ~ 불투명1.0)
2. 색상 지정 관련 속성
- `border`
- `background-color`
- `color`
- `opacity`(전체 투명도)

### 글꼴 -(4/15)
- 글꼴 관련 속성
    1. font-family 속성: 폰트 지정
        - 값: `<폰트패밀리>`(ex.serif, sans-serif 등), `<일반폰트명>`(구체적인 폰트명)
    2. font-size 속성: 글자크기 지정
        - 값: `<절대크기 키워드>`(xx-small, x-small, small, medium, large, x-large, xx-large), `<상대크기 키워드>`(smaller, lager), `<절대크기 단위>`(cm, mm, in, pt, pc, px), `<상대크기 단위>`(em, ex, %)
    3. font-style 속성: 글자모양 지정
        - 값: `normal`, `italic`, `oblique`(기울임)
    4. font-variant 속성: 작은 크기의 대문자로 변환여부
        - 값: `normal`, `small-caps`(작은대문자)
    5. font-weight 속성: 글자 굵기 지정
        - 값: `normal`(=400), `bold`(=700), `bolder`, `ligther`, `100~900 까지 100단위 숫자`
    6. font 속성: 글꼴 관련 속성들을 한번에 지정
        - `.style{ font: [weight값 | style값 | variant값] size값[/ lineheight값] family값; }`
    7. @font-face 규칙: 사용자 정의 글꼴 사용(TTF,OTF,WOFF등 가능)
        - `@font-face{ font-family:"사용자지정글꼴명"; src: url("글꼴의경로와파일명")}`
### 텍스트 -(5/15)
- 텍스트 줄맞춤 관련 속성
    1. letter-spacing, word-spacing 속성: 문자 사이의 간격/단어 사이의 간격
        - 값: <u>`normal`</u>, `<길이>`(음수가능->서로 겹침O)
    2. line-height 속성: 문장의 줄간격
        - 값: <u>`normal`</u>, `<길이>`, `<숫자>`(현재글꼴크기 기준), `%`(현재 글꼴크기 기준 백분율)
    3. text-align 속성: 텍스트 수평방향 정렬방식
        - 값: <u>`left`</u>, `right`, `center`, `justify`(양쪽 정렬)
    4. text-align-last 속성: 텍스트 마지막줄 정렬방식
        - 값: <u>`auto`</u>(=left), `right`, `center`, `justify`, `start`(라인 시작부분에), `end`(라인 끝부분에)
    5. text-indent 속성: 텍스트블록 첫번째줄 들여쓰기
        - 값: `<길이>`(기본값=0), `%`(부모요소 폭 기준 백분율)
    6. text-transform 속성: 텍스트 영어알파벳 표기방식
        - 값: <u>`none`</u>, `capitalize`, `uppercase`, `lowercase`
- 텍스트 장식 관련 속성
    1. text-decoration-line 속성: 텍스트 장식 라인의 종류 (1개 이상 동시 사용O)
        - 값: <u>`none`</u>, `underline`, `overline`, `line-through`(취소선)
    2. text-decoration-color 속성: 선 색상
        - 값: `<색상>`
    3. text-decoration-style 속성: 선 스타일
        - 값:<u>`solid`</u>, `double`, `dotted`, `dashed`, `wavy`
    4. text-decoration-thickness 속성: 선 두께
        - 값: `auto`, `<길이>`, `%`(1em에 대한 백분율)
    5. \*text-decoration 속성: 일괄지정(text-decoration-*)
        - 순서: line(필수)-color-style-thickness(공백으로 구분)
- 텍스트 효과 관련 속성
    1. text-shadow 속성: 텍스트에 그림자효과 (양수:오른쪽/아래쪽, 음수:왼쪽/위쪽)
        - 값:<u>`none`</u>, `<수평위치>`, `<수직위치>`, [`<흐림정도>`(기본값=0)], [`[<색상>]`(기본값=검정)]
        - 하나의 텍스트에 여러 그림자 효과O(콤마로 구분)
    2. text-overflow 속성: 박스영역을 벗어난 텍스트의 표시방식
        - 값: <u>`clip`</u>(표시X), `<ellipsis>`("..."표시)
    3. word-wrap 속성: 단어가 요소폭을 넘을때 줄바꿈시 단어 분리 여부
        - 값: <u>`normal`</u>(수행X), `<break-word>`
    4. vertical-align 속성: 요소의 수직 정렬 방식
        - 값: `<길이>`(음수O), `%`(음수O, 줄간격속성 기준 백분율),\
        <u>`baseline`</u>(부모요소의 기본줄에 맞춤), `<sub>`(아래첨자), `<super>`(위첨자),\
        `<top>`(해당줄 가장높은요소의 상단), `<middle>`(부모요소중앙), `<bottom>`(해당줄 가장낮은요소의 하단),\
        `<text-top>`(부모요소 폰트의 상단), `<text-bottom>`(부모요소 폰트의 하단)
### 리스트 -(6/15)
- 리스트 관련 속성
1. list-style-type 속성: 항목마커 종류
    - ul요소
        - 값: `none`, <u>`disc`</u>, `<circle>`, `<square>`
    - ol요소
        - 값: `<decimal>`, `<lower-alpha>`, `<upper-alpha>`, `<lower-roman>`,`<upper-roman>`
2. list-style-position 속성: 항목마커 위치
    - 값: <u>`outside`</u>, `<inside>`
3. list-style-image 속성: 이미지를 항목마커로 지정
    - 값: <u>`none`</u>, `url(이미지파일)`
4. \*list-style 속성: 일괄지정(list-style-*), 생략된 속성값은 기본값 사용

### 테이블 -(7/15)
- 테이블 관련 속성
1. border 속성: table,td,th요소의 테두리 지정 속성
    - `border: <두께> <스타일> <색상>`
2. table-layout 속성: 셀안의 내용 크기에 따른 셀크기 변화여부
    - 값: <u>`auto`</u>(자동결정), `<fixed>`(내용 크기에 무관, 테이블/셀 폭에 의해서만 결정)
3. border-collapse 속성: 테이블 테두리와 셀 테두리를 합칠지 여부
    - 값: <u>`separate`</u>(분리해 표시), `collapse`(하나로 합쳐 표시)--border-spacing과 empty-cell속성은 무시됨
4. border-spacing 속성: 인접 셀 테두리 간 간격
    - 값: `<수평거리><수직거리)`(기본값=0.0)--하나만 지정되면 나머지하나는 동일.
5. caption-side 속성: 테이블 캡션의 위치
    - 값: <u>`top`</u>(테이블상단), `<bottom>`(테이블하단)
6. empty-cells 속성: 빈 셀 테두리/배경 표시 여부
    - 값: <u>`show`</u>, `<hide>`

### 박스모델 -(8/15)
- css박스모델: 모든 HTML요소를 사각형형태 박스로 취급(레이아웃 구성 개념)->css로 각 박스의 위치/크기/색상 지정
1. 기본 속성
    - margin 속성: 박스 외부 여백(테두리 밖)
        - 일괄지정: 속성값 개수 => 4(상 우 하 좌), 3(상 좌우 하), 2(상하 좌우), 1(상하좌우) (기본값=`<길이>`0)
        - 개별방향 지정: margin-(top,right,bottom,left)
            - 값: `<길이>`, `%`(요소 폭 기준), `auto`
        - 인접한 두 요소의 bottom-top(수직여백)은 통합(더 큰 값으로)
    - padding 속성: 테두리 안과 요소 내용 사이의 여백 
        - 일괄지정: 속성값 개수 => 4(상 우 하 좌), 3(상 좌우 하), 2(상하 좌우), 1(상하좌우) (기본값=`<길이>`0)
        - 개별방향 지정: padding-(top,right,bottom,left)
            - 값: `<길이>`, `%`
2. 요소 크기 속성
    - width, height 속성: 요소 콘텐츠 표시영역 폭/넓이
        - 값: `<길이>`, `%`, <u>`auto`</u>
        - 속성이 하나만 지정->나머지 하나는 자동 결정
    - 크기제한(min-width, min-height, max-width, max-height 속성): 콘텐츠영역 최소폭/높이, 최대폭/높이 지정
        - 값: `<길이>`, `%`, <u>`none`</u>(미지정)
3. 요소 위치 속성
    - position 속성: 요소 배치의 기준 위치 
        - 값:
            - <u>`static`</u>: 페이지의 흐름에 따른 현재위치->별도 위치 지정불가
            - `<absolute>`: 브라우저 왼쪽상단 모서리 기준
            - `<relative>`: 해당요소의 현재위치를 (0,0)기준
            - `<fixed>`: 브라우저 윈도우 기준으로 고정->스크롤해도 변화X
    - top, bottom, left, right 속성: 요소를 포함하는 박스의 각 변으로부터 요소까지의 거리 지정
        - 값: <u>`auto`</u>, `<길이>`, `%`(포함블록 기준)
    - z-index 속성: 요소간 겹쳐지는 순서
        - 값: <u>`auto`</u>, `<정수>`(클수록 앞에위치, 음수O)
        - 동작조건: `position: relative | absolute | fixed`
4. 화면 표시 속성
    - display 속성: 요소 표시 방식(=사용할 박스 종류)
        - 값: `none`(표시x), `inline`(인라인요소취급, 줄바꿈x), `block`(블록요소취급, 줄바꿈o), `inline-block`(인라인수준의 블록취급=박스는 인라인, 내부는 블록), `list-item`(목록의 항목) 
    - overflow 속성: 요소박스를 벗어나는 경우의 처리방법
        - 값: <u>`visible`</u>, `hidden`, `clip`, `scroll`, `auto`
        - height 속성이 지정된 블록요소만 적용가능
        - overflow-x, overflow-y 속성: 수평/수직방향 구분 지정
    - visibility 속성: 요소박스 화면표시 여부
        - 값: <u>`visible`</u>, `hidden`(화면표시x,페이지공간차지o), `collapse`(hidden과 동일;테이블요소->행/열 제거)
5. 플로팅 속성
    - float 속성: 콘텐츠의 일반적 흐름x => (부모요소영역기준)왼쪽/오른쪽에 배치
        - 값: <u>`none`</u>, `left`, `right`
    - clear 속성: float속성 영향을 받는 요소의 흐름 해제->바로 아래쪽에 요소 배치
        - 값: <u>`none`</u>, `left`, `right`, `both`
6. 박스 크기 속성
    - box-sizing 속성: width/height속성으로 폭/높이 지정시 패딩과 테두리 포함여부
        - 값: <u>`content-box`</u>(포함X), `border-box`(포함o)
    - resize 속성: 사용자의 요소박스 크기조절 여부
        - 값: <u>`none`</u>, `both`(폭,높이), `horizontal`(폭), `vertical`(높이)
        - `overflow: hidden|scroll|auto` 와 함께 사용

### 테두리 -(9/15)
- border 관련속성
    - border[-top|right|bottom|left][-width|color]: 방향별 border속성
    - 속성값 개수에 따른 적용방향O(4:상 우 하 좌, 3:상 좌우 하, 2:상하 좌우, 1:상하좌우): width, style, color
    1. border-width 속성: 테두리 굵기
        - 값: <u>`medium`</u>, `thin`, `thick`, `<길이>`
    2. border-style 속성: 테두리 스타일
        - 값: <u>`none`</u>, `hidden`, `dotted`, `dashed`, `solid`, `double`, `groove`(오목한선), `ridge`(볼록한선), `inset`(안으로 파인 st), `outset`(밖으로 나온 st)
    3. border-color 속성: 테두리 색상
        - 값: `<색상>`(<u>검정</u>), `transparent`
    4. \*border 속성: border-*을 한번에 지정(style은 필수, width-style-color순서)
    5. border-radius 속성: 요소박스 테두리를 둥글게
        - 값: `<X_1,X_2,X_3,X_4:길이|%>[/<Y_1,Y_2,Y_3,Y_4:길이|%>]`
        - 각 모서리별 속성 지정 가능(border-<모서리>-radius 속성)
            - 1: top-left
            - 2: top-right
            - 3: bottom-right
            - 4: bottom-left
    6. box-shadow 속성: 박스요소에 그림자스타일 지정
        - 값: <u>`none`</u>, `<그림자>`
            - <그림자>::*`<수평위치> <수직위치>`* `[<흐림정도>`*`<퍼진거리>`* `<색상> inset]`
                - *기울임체*: 음수가능
                - <수평위치>,<수직위치>: 필수지정
                - <색상>: 기본값=검정색
                - inset: 박스 안쪽 그림자로.
### 배경 -(10/15)
1. background-color: 요소의 배경색
    - 값: <u>`transparent`</u>, `<색상>`
2. background-image: 요소의 배경이미지
    - 값: <u>`none`</u>, `url(파일)`
    - 여러개 이미지 지정 가능(콤마 구분자, 계층형성o)
3. background-repeat: 배경이미지의 반복여부
    - 값: <u>`repeat`</u>, `repeat-x`, `repeat-y`, `no-repeat`,\
    (배경영역공백에서 잘려보이지 않도록)`space`(공백조정), `round`(이미지크기조정)
4. background-attachment: 스크롤에 따른 배경이미지 움직임여부
    - 값: <u>`scroll`</u>(콘텐츠와 함께 스크롤), `fixed`(뷰포트에 고정), `local`(요소 내부에서 스크롤->요소 내 콘텐츠와 함께 스크롤)
5. background-position: 배경이미지 시작위치
    - 값: <u>`<xpos> <ypos>`</u>
        - xpos: left | center | right | % | px
        - ypos: top | center | bottom | % | px
        - 하나만 지정 시 나머지는 center로 취급
    - 기본위치: 요소 왼쪽 상단모서리(0% 0% or left top)
6. background-origin: 배경이미지 시작 기준위치
    - 값: <u>`padding-box`</u>, `border-box`, `content-box`
7. background-size: 배경이미지 크기
    - 값: <u>`auto`</u>, `<폭_길이, 높이_길이>`, `<폭_%, 높이_%>`, `cover`(이미지 폭,길이 중 짧은 것 기준 영역채우기=원래 크기비율 무시), `contain`(이미지 폭,길이 중 긴 것 기준 영역채우기=원래 크기비율 유지)
8. background-clip: 배경속성(색상,이미지) 적용영역
    - 값: <u>`border-box`</u>, `padding-box`, `content-box`
9. \*background: background-*의 일괄지정
    - -color -image -position / -size -repeat -origin -clip -attachment
        - 각 속성 생략 가능
        - -size 속성 사용시 "/" 필수
        - -image의 이미지파일 여러개(콤마로 구분) && -color 속성 지정 시 => -color 속성을 리스트 맨 뒤에

### 그라데이션 -(11/15)
- 그라데이션: 두 개 색상 사이에서 점진적 변화
- 사용법: `{background: {linear|radial|conic}-gradient(...)}`
1. 선형 그라데이션
    - `linear-gradient(진행방향, 색상1, 색상2, ...)`
        - 진행방향: `to 키워드` or `각도(deg)`
        - `키워드`: top(0deg), <u>bottom(180deg)</u>, right(90deg), left(270deg),\
        top right(45deg), bottom right(135deg), top left(315deg), bottom left(225deg)
        - 색상 2개 이상인 경우 `색상1[위치백분율%], 색상2[위치백분율%][, 색상3[위치백분율%], ...]`
    - `repeating-linear-gradient(진행방향, 색상1, 색상2, ...)`: 마지막 `색상[위치백분율%]`표기를 기준으로 반복횟수 결정
2. 방사형 그라데이션
    - `radial-gradient(<shape> <size> at <position>, 색상1, 색상2, ...)`
        - shape: <u>ellipse</u>(타원), circle
        - size: closest-side(가까운 모서리까지), farthest-side(먼 모서리까지), closest-corner(가까운 꼭짓점까지), <u>farthest-corner</u>(먼 꼭짓점까지)
        - position(중심의 위치): px, % (기본값=center)
    - `repeating-radial-gradient(<shape> <size> at <position>, 색상1, 색상2, ...)`: 마지막 `색상[위치백분율%]`표기를 기준으로 반복횟수 결정(size기준까지의 반복횟수를 의미함.)
3. 원뿔형 그라데이션 
    - `conic-gradient([from 시작각도][ at 중심위치,] 색상1[위치각도deg], 색상2[위치각도deg], ...)`
        - 시작각도: deg (기본값=0deg=12시방향)
        - 중심위치: %, px (기본값=50% 50%)
    - `repeating-conic-gradient([from 시작각도][ at 중심위치,] 색상1[위치각도deg], 색상2[위치각도deg], ...)`: 마지막 `색상[위치각도deg]`표기를 기준으로 반복횟수 결정

### 변형 -(12/15)
- 요소 형태,크기,위치 변경 효과
    - 이동translate, 회전rotae, 크기변경scale, 기울임skew
- 관련 속성
    1. transform: 요소에 변형지정(변형함수)
        - 값: <u>`none`</u>, `<변형함수>+`
        - 변형함수(다중 변환 지정 가능)
            - 이동: 요소 중앙점 기준 (음수O)
                - translate(x,y), translate3d(x,y), translateX(x), translateY(y), translateZ(z)
            - 크기: 
                - scale(x,y), scale3d(x,y), scaleX(x), scaleY(y), scaleZ(z)
            - 회전: rotate(angle), rotate3d(x,y,z,angle), angleX(angle), angleY(angle), angleZ(angle)
            - 기울임: skew(x-angle, y-angle), skewX(x), skewY(y)
            - 변환행렬 matrix(scaleX(),skewY(),skewX(),scaleY(),translateX(),translateY()): 이동, 크기확대축소, 회전, 기울임을 한번에 지정
                - 값: 소수포함 정수(기본값= 1,0,0,1,0,0)
                - 단위 입력 불필요(크기:배수, 기울임:각도, 이동:픽셀 로 간주)
                - matrix3d(n,n,n,n,n,n,n,n,n,n,n,n,n,n,n,n)
    2. transform-origin: 요소 박스에 대한 변형 기준점
        - 값: `<X축값> <Y축값> [<Z축값>]`
            - X축값: left(=0%), center(=50%), right(=100%), <길이>, %
            - Y축값: top(=0%), center(=50%), bottom(=100%), <길이>, %
            - Z축값: <길이>(기본값=0)
        - transform 속성과 함께 사용!
    3. perspective: 3차원 공간 상 해당 요소와 관측점 거리 지정 (자식요소에 원근감 적용됨.)
        - 값: <u>`none`</u>, `<길이>`
            - 길이가 작을수록 더 기울여지는 정도가 커보임
    4. perspective-origin: 원근 조절 시 X축,Y축에 대한 기준점 지정
        - 값: `<길이>`, `%`, `left`, `center`, `right`, `top`, `bottom`
        - perspective 속성과 함께 사용!
    5. backface-visibility: 요소 뒷면 표시 여부 지정
        - 값: <u>`visible`</u>, hidden

### 전환 -(13/15)
- 어떤 이벤트에 대해 지정 시간내 속성 상태->다른 상태로 부드럽게 변화
    - 점진적 변화: 애니메이션 효과 부여
- 방법: 어떤 이벤트에 부여할지 -> 속성의 처음상태와 최종상태 지정
- 속성
    - transition-property속성: 전환이 적용될 속성이름 나열(구분자: 콤마)
    - transition-duration속성: 전환 진행 시간 지정
    - transition-delay속성: 전환 효과 시작 전 대기시간 지정(값: `초(s)`,`밀리초(ms)`)
    - transition-timing-function속성: 전환 진행 속도의 형식 지정
        - 값: <u>`ease`</u>(slow->fast->slow), linear, ease-in(slow->fast), ease-out(fast->slow), ease-in-out,\
        steps(n,start|end)--(전환진행을 나눠보여줄 간격개수n, 각간격에서 전환효과가 발생하는 지점)--, step-start, step-end, cubic-bezier(n,n,n,n)
- 여러 속성 전환 지정 시: transition의 각 속성에서 콤마로 구분하여 대응
- 전환 일괄 지정: transition속성
    - `-property값 -duration값 -timing-function값 -delay값` 1개 이상을 콤마로 구분하여 지정
    - 생략가능: timing-function값, transition-delay값
- 전환 부여 가능 속성에 제한o
- 변형(transform 등)을 활용한 전환O

### 애니메이션 -(14/15)
- 키 프레임: 애니메이션 구성하는 움직임의 키가 되는 프레임(각 프레임 연결->동작 구성)
```css
@keyframes 애니메이션이름{
    from { 애니메이션의 시작시점에서 CSS스타일 }
    [ percentage { 해당 중간시점에서 CSS스타일 } 
     ... ]*
    to { 애니메이션 종료시점에서 CSS스타일 }
}
```
- 애니메이션 속성
    - animation-name: 키프레임 이름 (값: <u>none</u>, <키프레임이름>)
    - animation-duration: 한 사이클 실행시간 (값: <u>0</u>, <초>)
    - animation-delay: 시작 전 대기시간 (값: <u>0</u>, <초>)
    - animation-timing-function: 진행속도 형식 (값: ease, linear, ease-in, ease-out, ease-in-out)
    - animation-iteration-count: 반복횟수 (값: <횟수>--기본값=1, infinite)
    - animation-direction: 진행방향 (값: <u>normal</u>, reverse, alternate, alternate-reverse)
    - animation-play-state: 실행상태or일시정지상태 (값: <u>running</u>, paused)
    - animation-fill-mode: 실행 이전/이후의 스타일 유지여부 (값: <u>none</u>, forwards, backwards, both)
    - 일괄지정(animation): (필수) -name, -duration,\
    -timing-function, -delay, -iteration-count, -direction, -fill-mode, -play-state
- 변형(rotate 등)을 활용 가능

### 다단 -(15/15)
- 다단 지정
    - columns: `column-width값 column-count값`
        - -width: 단의 폭(값: <u>auto</u>, <길이>)
        - -count: 단의 개수(값: <u>auto</u>, <개수>)
        - 두개가 동시에 auto 불가, 1개만 지정시 나머지는 요소의 폭에 따라 자동지정
- 속성
    - column-gap: 단과 단 사이 간격 (값: <u>normal</u>--기본값=1em, <길이>)
    - column-rule: 단과 단 사이 구분선 관련속성 일괄지정
        - -width: 굵기 (값: thin, medium, thick, <길이>)
        - -style: 스타일 (값: none, hidden, dotted, dashed, solid, double, groove, ridge, inset, outset)
        - -color: 색상 (값: <색상>, transparent)
    - column-span: 요소가 얼마나 많은 단을 차지해서 표시할지 (값: <u>none</u>, all)
    - column-fill: 각 단을 채우는 콘텐츠 양 조절-> 각 단의 높이 차이 최소화 여부 (값: <u>balance</u>, auto)

<!-- *** -->
## Application(~3)
### example review
### confusing point
- perspective: 3차원 공간 상 해당 요소와 관측점 거리 지정 (자식요소에 원근감 적용됨.)
    - 값으로 지정한 길이가 작을수록 더 기울여지는 정도가 커보임 -> 왜?
### extension point(implement)
- css 너무 지엽적으로 복잡하다
    1. 왜 이래야만 할까?
    2. 더 편리한 툴이 뭐가 있고 그건 어떻지?
## Reference(bookmark)


<!-- ------------------------------ -->
# Chapter 4. javascript
>js개요/선언/변수,자료형/연산자/조건문,반복문\
/함수/객체/이벤트
### Outline
### Flow Diagram
<!-- *** -->
## Main Concepts
### 자바스크립트개요 -(1/8)
- HTML5의 공식적 스크립트 언어
    - 스크립트: 프로세서가 아닌 다른 프로그램에 의해 번역or수행되는 프로그램/명령어나열
    - 사용자-페이지 간의 상호작용o(생동감, 동적웹페이지)

### 선언 -(2/8)
- 선언방식
    1. 내부 스크립트 방식: 문서에서 `<srcipt>`요소 사용(종료 태그O)
    2. 외부 스크립트 방식: 외부파일(*.js), `<srcipt>`요소 안 src속성으로 불러오기
    3. 인라인 스크립트 방식: HTML요소 내부 이벤트속성값으로 코드삽입
- 주석
    - 한줄 주석: `//`
    - 여러줄 주석: `/* */`

### 변수와 자료형 -(3/8)
1. 변수: 프로그램에서 다루는 데이터 저장 공간
    - 선언: 저장할 적정크기의 공간 확보
    - 자료형(데이터 종류)와 식별자(구별할 이름)이 필요
2. 식별자: 변수이름or함수이름(프로그래머가 만든 단어)
    - 지정규칙:\
    (필수) 첫글자=영문자,언더바(\_),\$로 시작, 두번째~=영문자,숫자,언더바(\_),\$, 대소문자구분, 키워드(let,if,for,function,...)사용불가\
    (권장) 카멜케이스, 의미내포&충분한길이의 단어
3. 자료형
    - 숫자형: 정수(10진수:0으로 시작x, 8진수:0으로 시작, 16진수:0x 또는 0X로 시작), 실수(소수점 포함 숫자)
    - 문자열: 따옴표(")로 묶인 일련문자
    - 논리형: 참(true)/거짓(false)
    - null형: 내용이 전혀없는 공백값(변수 초기화 시 사용)
4. 변수선언
    - 명시적 선언 없이 사용 가능(자동 생성, 값 대입 시 자료형 결정)
    - 명시적 선언 시: `let 변수명` => 자료형 무관(저장되는 값에 따라 결정)
        - 함수 내에서 let으로 선언 -> 지역변수
        - 함수 밖 or let없이 선언 -> 전역변수
5. 상수선언
    - 예약어 const 사용
    - 값의 변경/재지정 불가
    - 사용범위: 선언된 블록 내

### 연산자 -(4/8)
1. `(),[]`
2. `!,++,--,~`
3. `**`, `*,/,%`
4. `+,-`, `>>,<<,>>>`
5. `>,<,>=,<=`, `>==, !=, ===, !=`
6. `&`, `^`, `|`, `&&`, `||`, `?:`
7. `=, +=, -=, /=, *=, %=`

### 조건문과 반복문 -(5/8)
1. 조건문
    ```js
    //if문
    if(조건식)
        문장

    if(조건식) {
        문장...;
    }
    //if-else문
    if(조건식) {
        문장1;...
    }
    else {
        문장a;...
    }
    //switch문
    switch(표현식){
        case 값1:문장;...
            break;
        case 값2:문장;...
            break;
        ...
        default:문장;...
    }
    ```
2. 반복문
    ```js
    //for문
    for(초기식;조건식;증감식){
        문장...;
    }
    //while문
    while(조건식){
        문장...;
    }
    //do-while문
    do{
        문장...;
    } while(조건식)
    ```
### 함수 -(6/8)
- 보통 head부분에서 정의
    - 정의
        ```js
        function 함수명([매개변수1, ... ,매개변수n]){
            문장;
            ...
            [return 반환값;]
        }
        ```
    - 호출: `[ 변수 ]=함수명([매개변수1, ..., 매개변수n];`
- 무명함수: 함수명 지정X로 선언된 함수 (함수를 변수에 저장, 나중에 한번만 호출해 사용)
    - 정의
        ```js
        let 변수 = function([매개변수1, ... ,매개변수n]){
            문장;
            ...
            [return 반환값;]
        }
        ```
    - 호출: `변수([매개변수1, ..., 매개변수n]);`
- 콜백함수: 매개변수로 전달되는 함수 
    - (완결성이 있는 단위작업을 반복해야할때 단순출력과 로직을 분리하는 용도?)
    ```js
        function runNTimesSum(n, callback){
            for(let i=1; i<=n; i++){
                document.write("1+...+"+i+"까지의 합:")
                callback(i)
            }
        }
        n_sum(10, function(n){
            let sum=0;
            for(let i=1; i<=n; i++){
                sum+=i;
            }
            document.wirte(sum+"<br/>");
        })
    ```
- 대화상자 관련 함수
    1. 알림창 alert("메시지"): "확인" 버튼을 갖는 메시지 창을 띄우는 함수 (경고 같은 안내메세지)
    2. 확인창 confirm("메시지"): 메세지를 보여주고 "확인" 또는 "취소"버튼을 선택하도록 하여 이 결과값에 따라 다음 작업 수행
    3. 입력창 prompt("대화상자설명", 기본값): 사용자에게 키보드로 데이터 입력받을때 사용
### 객체 -(7/8)
- 자바스크립트 작업 대상(기본자료형)
- 실세계 유무형의 것 모델링 <- 변수,함수를 그룹화
    - 속성: 객체 특성/상태를 표현-데이터
    - 메서드: 객체 관련 작업 처리-함수(데이터 동작 규칙/방법)
- 객체 생성: `객체변수 = new 객체명([매개변수리스트]);`
- 객체 사용: `객체변수.속성`, `객체변수.메서드([매개변수리스트]);`
1. 내장 객체(코어객체)
    - String 객체: 문자모양 지정or문자열 다룸
        - new연산자 사용 or 문자열을 변수에 할당 시 자동생성
        - 속성: length(문자열 길이 반환)
        - 메서드
            1. 글꼴 관련: big(), small(), bold(), italics(),\
            strike(), sub(), sup(), fontcolor("색상"), fontsize(1~7숫자)
            2. 문자열처리 관련: toLowerCase(), toUpperCase(), substring(n1,n2), slice(n1,n2), substr(i,n),\
            charAt(i), indexOf("문자"), lastIndexOf("문자"), concat("문자열"),\
            replace('문자열1','문자열2'), split("구분자",개수), charCodeAt(i), String.fromCharCode(유니코드번호), " 공백포함문자열 ".trim()
            3. 위치이동 관련: "문자열".link("URL"), "문자열".anchor("이름") --URL은 #이름 or 외부주소 형태
    - Array 객체: 배열 사용을 위한 객체
        - 생성: `배열객체변수 = new Array()`, `배열객체변수 = new Array(배열크기)`, `배열객체변수 = new Array(배열요소1, ... ,배열요소n)`
        - 속성: length(배열 길이 반환)
        - 메서드: join(), indexOf("데이터"), push("데이터"), shift(),\
        reverse(), concat(배열2), sort(), pop(), slice(i,j)
    - Date 객체: 사용자 시스템 날짜/시간 관리 객체
        - 생성: `객체변수 = new Date()`, `객체변수 = new Date(년,월,일[,시[,분[,초[,밀리초]]])`
        - 메서드: 
            - get/set: __FullYear(), __Month()--월(0:1월~11:12월), __Date(), __Day()--요일(0:일요일~6:토요일), __Hours(), __Minutes(), __Seconds(), __Milliseconds(), __Time()--70.01.01 0시부터의 시간을 1/1000초 단위
            - 표시형식지정: toGMTString(), toLocaleString(), toString()
    - Math 객체
        - 속성: PI(파이), SQRT2(루트2)
        - 메서드
            - abs(n), acos(n), asin(n), atan(n), atan2(x,y), cos(n), tan(n)
            - ceil(n), floor(n), exp(n), log(n)
            - max(n,m), min(n,m), pow(n,m)
            - random(), round(n), sin(n), sqrt(n)
2. 사용자 정의 객체
- 프로토타입 기반 언어
- 일반함수와 동일형식의 생성자함수 이용
    ```js
    function 프로토타입명(매개변수1, ..., 매개변수n){
        this.속성1=매개변수1;
        ...
        this.속성n=매개변수n;
        this.메서드1=function() { ... }
        ...
        this.메서드n=function() { ... }
    }
    ```

### 이벤트 -(8/8)
- 사용자가 어떠한 동작수행 or 문서/브라우저 상태가 변할때 발생하는 특정 사건/동작
    - 이벤트: 사건/동작
    - 이벤트 리스너: 처리내용 (js코드로 작성, 호출되어 수행)
- 주요 이벤트
    1. 윈도우: `load`, `unload`(문서 및 이미지 로딩 완료/문서 닫을때)
    2. 포커스: `focus`, `blur`(객체가 포커스를 가질때/잃을때)
    3. 폼: `change`(input,select요소등의 값이 변경 시), `select`(입력된 텍스트를 사용자가 선택 시), `submit`(submit버튼을 눌러 폼을 전송 시), `reset`(reset버튼으로 초기화 시)
    4. 키보드: `keydown`(누르는 순간), `keyup`(누른 키를 놓는 순간), `keypress`(눌렀다 놓는 순간)
    5. 마우스: `click`(클릭), `dbclick`(더블클릭), `mousedown`(마우스버튼 누르는 순간), `mouseup`(눌러진 마우스버튼을 놓는 순간), `mousemove`(마우스가 객체위에서 움직이는 동안), `mouseover`(커서가 객체영역 안으로 들어가는 순간), `mouseout`(커서가 객체영역 밖으로 벗어나는 순간)
- 이벤트 리스너 작성방법
    1. `<요소명 속성="값" ... on이벤트명="자바스크립트코드">`
    2. `addEventListener("이벤트명", 실행함수명);`
- 이벤트 객체: 이벤트와 관련된 다양한 정보o 객체
    - 전달하는 함수에서의 매개변수명: event 사용
        - 전달받는 함수에서의 매개변수명->원하는 이름 지정
    - 정보?: 이벤트 종류에 따라 다름
        - ex. 마우스클릭(커서위치, 클릭한마우스버튼 등), 키보드조작(눌러진 키이름,문자열, alt/shift/ctrl과 함께press 여부 등)
<!-- *** -->
## Application
### example review
### confusing point
### extension point
(implement)
## Reference(bookmark)


<!-- ------------------------------ -->
# Chapter 5. 문서 객체 모델, 브라우저 객체 모델
>문서객체모델/브라우저객체모델
### Outline
### Flow Diagram
<!-- *** -->
## Main Concepts
### 문서 객체모델 -(1/2)
1. Document Object Model
    - 문서객체: html 각 요소를 js에서 사용가능하도록 객체로 만듦
    - DOM: 브라우저가 표준 html문서에 접근가능하도록 정의한 표준모델 (HTML문서를 분석/표시하는 방식)
    - 생성: 문서 적재->브라우저가 정적으로 DOM생성
        - 계층적 구조를 갖는 DOM트리 형태
        - js이용 => 프로그램 실행 중 웹문서 내용/구조/스타일에 작업가능=즉각적으로 화면 반영o
2. DOM객체 접근방법
    - id속성 이용: `getElementById("아이디명")` --아이디명을 id속성값으로 갖는 오직 하나의 요소 선택
    - 요소명 이용: `getElementByTagName("요소명")` --지정 요소명 모두를 선택, 리스트형태로 반환
    - 클래스명 이용: `getElementByClassName("클래스명")` --클래스명을 class속성값으로 갖는 모든 요소, 리스트형태로 반환
3. 요소 수정(내용/속성/스타일)
- 내용
    - innerHTML속성: HTML형식으로 요소내용을 다룰때\
    `DOM변수 = document.getElementById("아이디명");`
        - get: `변수 = DOM변수.innerHTML`
        - set: `DOM변수.innerHTML = "값"`
    - textContent속성: 
        - get: HTML요소,속성을 제외하고 텍스트내용만 가져오는 경우(`변수 = DOM변수.textContent`)
        - set: HTML요소를 일반텍스트로 취급해 요소내용지정(`DOM변수.textContent = "값"`)
- 속성
    - HTML요소 속성을 그대로 사용, 새로운 값 할당
    - `document.getElementById("아이디명").속성=값`
- 스타일
    - css스타일 동적수정O: `document.getElementById("아이디명").style.CSS속성명=값`
    - css속성명이 하이픈포함->하이픈을 없애고 카멜케이스로 표기
4. 새로운 요소 삽입
    - 텍스트내용이 있는 요소
        1. 요소 DOM객체 생성: `let 새요소변수=document.createElement("삽입요소명")`
        2. 요소에 텍스트 내용: `새요소변수.innerHTML="텍스트내용"`
            - or `let 텍스트변수=document.createTextNode("텍스트내용")` `새요소변수.appendChild(텍스트변수)`
            + CSS스타일추가: `새요소변수.style.CSS속성명=값`
        3. 새요소의 부모요소를 찾아 자식요소로 연결: `let 부모요소변수=document.getElementById("부모요소아이디명")` `부모요소변수.appendChild(새요소변수)`
    - 속성이 있는 요소
        1. 요소 DOM객체 생성: `let 새요소변수=document.createElement("삽입요소명")`
        2. [요소의 속성,속성값 추가]+: `새요소변수.setAttribute("속성명","속성값")`
            - or `let 속성변수=document.createAttribute("속성명");` `속성변수.value="속성값";` `새요소변수.setAttributeNode(속성변수);`
        3. 새요소의 부모요소를 찾아 자식요소로 연결: `let 부모요소변수=document.getElementById("부모요소아이디명");` `부모요소변수.appendChild(새요소변수)`
5. 기존 요소 삭제
- DOM구조에서 삭제하려는 노드의 부모노드 찾기->removeChild()메서드 적용
- `부모요소.removeChild(삭제하려는요소)`

### 브라우저 객체모델 -(2/2)
- Browser Object Model: 브라우저를 객체로 표현한 것
    - window (최상위객체)
        - document (HTML 문서를 표현, DOM객체의 최상위객체)
        - history (사용자가 방문한 URL)
        - location (현재 URL에 대한 정보유지)
        - navigator (현재 사용중인 브라우저 정보관리)
        - screen (스크린 장치에 대한 정보관리)
    - 특정객체의 메서드/속성 사용: `window. ... .객체.속성`, `window. ... .객체.메서드()`
- window 객체: 창에 대한 전반적인 모든상황 제어(최상위객체)
    - 각 윈도우마다 하나의 window객체 생성
    - 메서드
        - open(), close(): 새로운 윈도우를 열기/열린 창을 닫기
            - `open(연결할URL, target창, 윈도우속성리스트)`
                - target창: (값: `_blank`, `_self`, `창의 이름`)
                - 속성리스트: `"속성=값,속성=값,..."`
                    - location: y/n(주소표시줄 표시여부)
                    - status: y/n(상태표시줄 표시여부)
                    - menubar: y/n(메뉴바 표시여부)
                    - resizable: y/n(사용자에 의한 창크기조절 가능여부)
                    - height, width: px(윈도우 높이/너비)
                    - left, top: px(윈도우 표시될 x축/y축 위치)
        - setInterval("호출함수", 시간간격): 일정시간 간격 함수 무한반복호출
            - clearInterval(timerID): 무한히 반복되는 동작을 정지시키는 함수(`timerID=setInterval(...)`)
        - setTimeout("호출함수", 지연시간): 일정 시간 후 지정함수 수행
            - 지연시간: 코드실행까지의 시간, 1/1000초 단위
            - clearTimeout(timerID): 설정된 시간 전에 실행 해제하는 함수(`timerID=setTimeout(...)`)
        - alert(), confirm(), prompt(): 경고창, 확인창, 입력창
        - moveBy(x,y), moveTo(x,y): 창의 (상대/절대)위치 이동
        - resizeBy(w,h), resizeTo(w,h): 창의 (상대/절대)크기 조정

<!-- *** -->
## Application
### example review
### confusing point
### extension point
(implement)
## Reference(bookmark)

<!-- ------------------------------ -->
# Chapter 6. 캔버스
>캔버스개요/선,도형그리기/스타일지정\
/텍스트,이미지그리기/도형변환
### Outline
### Flow Diagram
<!-- *** -->
## Main Concepts
### 캔버스 개요 -(1/5)
1. 자바스크립트 코드를 이용해 웹브라우저에 그림 그리기 기능
    - 별도 프로그램 설치X
    - 단순 그림을 그리는 기능 + 그림 합성,변환,애니메이션과 같은 다양한 효과 표현
2. 방법
    1. canvas요소(컨테이너)를 사용해 그림영역 지정
    2. js코드 사용, 해당영역에 실제 그림 그리기
3. canvas 요소: 캔버스 영역(웹페이지에 그림이 그려지는 투명 사각형 영역) 지정
    - `<canvas id="아이디" width="폭" height="높이">canvas요소 미지원 브라우저에 출력할 내용</canvas>`
    - id속성 필수지정, width/height생략시 기본값=300px/150px
4. 캔버스 객체 & 캔버스 컨텍스트 객체
    - 캔버스 생성 -> 캔버스 객체 생성 -> 캔버스 컨텍스트 객체 생성 -> 실제 그림그리기 작업
        - 캔버스 객체 생성: `let cvs변수1=document.getElementById('canvas요소id속성값');`
        - 캔버스 컨텍스트 객체 생성: `let ctx변수2=cvs변수1.getContext('2d');`
        - 그림 그리기 작업: `ctx변수2.메서드()`, `ctx변수2.속성`
    
### 선과 도형 그리기 -(2/5)
1. 사각형 그리기: 시작점(x,y), 폭w, 높이h
    - 메서드
        - strokeRect(x,y,w,h): 테두리만 있는 사각형
            - 관련속성: strokeStyle, lineWidth, lineJoin, lineCap 
        - fillRect(x,y,w,h): 색이 채워진 사각형
            - 관련속성: fillStyle, grobalAlpha
        - clearRect(x,y,w,h): 지정 사각형 영역 지우
            - 주어진 영역을 rgba(0,0,0,0)으로 채움
2. 캔버스 그림그리기 방식
    - 호출 즉시: 호출시 어떤 단계도 거치지않고 바로 캔버스에 그림을 그림
        - 사각형 그리기(strokeRect(), fillRect(), clearRect())
        - 텍스트 그리기(strokeText())
        - 이미지 그리기(drawImage())
    - 패스 기반: 각 도형을 이루는 선(=서브패스)들의 집합
        - 과정
            1. 패스 초기화: beginPath()--이전 패스를 모두 지우고 새로운 패스 그림
            2. 다양한 메서드 사용해 패스 지정->선/도형 그림
                - moveTo(x,y): 주어진 점 시작으로 새로운 서브패스 만듦
                - lineTo(x,y): 바로 이전의 점과 현재의 점을 연결하는 선을 그림
                - rect(x,y,w,h): 사각형을 그림
                - arc(x,y,r,...): 원/원호를 그림
            3. 지정한 패스 닫기: closePath()--현재 패스를 닫음
            4. 선/도형 출력: stroke()/fill()--현재 패스에 있는 도형들을 선형태로/내부를 채워 실제 캔버스에 그림
3. 패스 기반 여러 도형 그리기
    1. 선 그리기: `ctx.beginPath();` -> `ctx.moveTo(x1,y1)` -> `ctx.lineTo(x2,y2)` -> `ctx.stroke()`
    2. 다각형 그리기
        - 방법1: 선을 그리는 lineTo()를 연속적으로 사용
        - 방법2: 선 그리기 메서드 + closePath()--맨 마지막 점과 첫번째 점(moveTo(x,y))을 연결하는 직선을 자동 추가하는 메서드
    3. 원/원호 그리기: arc(x,y,r,시작각도,종료각도,그리는방향)
        - 시작각도/종료각도: 라디안 단위(=각도*Math.PI/180), 3시를 기점으로 시계방향으로 부여
        - 그리는 방향: false(기본값, 시계방향), true(반시계방향)
    4. 부채꼴 그리기: `moveTo(x,y)` -> `arc(x,y,...)` -> `closePath()`

### 스타일 지정 -(3/5)
1. 선 스타일
    - 관련속성
        - strokeStyle: 선의 색상or그라데이션,패턴)
        - lineWidth: 선의 두께, 기본값=1px
        - lineJoin: 두 선이 만나 꺾이는 모서리부분 모양
            - 값: <u>`miter`</u>,`round`,`bevel`
        - lineCap: 선의 양쪽 끝부분 모양
            - 값: <u>`butt`</u>,`round`,`square`
    - 점선 그리기: setLineDash() --원하는 점선 패턴 설정
        - 파라미터:점선에서 선이 그려지는 부분과 그렇지 않은 부분의 길이(ex. `setLineDash([5,4,3,2])`--길이 14중 5를 칠하고 이후 4칸은 비우고 이후 3칸 칠하고 2칸은 비움.)
2. 채우기 스타일
    - 관련속성
        - fillStyle: 도형을 채우는 색상(그라데이션,패턴)
        - globalAlpha: 색상의 투명도(값: 투명0.0~불투명1.0)
3. 그라데이션 스타일: 두 가지 이상 색상이 점진적 변하는 효과
    - 선형 그라데이션: createLinearGradient(x1,y1,x2,y2)
    - 방사형 그라데이션: createRadialGradient(x1,y1,r1,x2,y2,r2)
        - r1은 시작점색깔의 상한, r2는 종료점색깔의 하한, (r2-r1)/2 값만큼 시작점색깔+중간점색깔, 중간점색깔+종료점색깔
    - 그라데이션 경계색 지정: addColorStop(오프셋,색상)
        - 오프셋: 경계색 위치(시작점0.0~끝점1.0)
    - 지정형식
        ```js
        ctx.beginPath();
        let 변수=ctx.createLinear[|Radial]Gradient(x1,y1[,r1],x2,y2[,r2]);
        변수.addColorStrp(시작점오프셋, 색상);
        변수.addColorStrp(중간점오프셋, 색상);
        변수.addColorStrp(종료점오프셋, 색상);
        ctx.fillStyle=변수;
        ```

4. 패턴 스타일: createPattern(이미지변수, 반복형식)
    - 채우기를 위한 이미지 패턴 생성
    - 반복형식: no-repeat, repeat-x, repeat-y, repeat
5. 그림자 스타일
    - 관련 속성
        - shadowColor: 그림자 색상(기본값: rgba(0,0,0,0))
        - shadowOffsetX: 대상 기준 그림자 수평오프셋(기본값=0)
        - shadowOffsetY: 대상 기준 그림자 수직오프셋(기본값=0)
        - shadowBlur: 그림자의 흐린정도(기본값=0)
6. 도형 합성
    - globalCompositeOperation속성: 도형이 그려지는 순서에 상관x 겹쳐지는 부분에 대한 다양한 처리
        - 속성값: <u>source-over</u>, source-atop, source-in, source-out, destination-over, destination-atop, destination-in, destination-out, lighter, copy, xor
            - source: 속성 지정 후에 그려진 도형 / destination: 속성 지정 전에 그려진 도형

### 텍스트와 이미지 그리기 -(4/5)
1. 텍스트 그리기
    - 관련 메서드
        - strokeText(텍스트,x,y[,maxWidth]): 지정된 위치(x,y) 기준 테두리만 있는 텍스트를 그림.(maxWidth: 텍스트가 주어진 크기에 맞게 조정)
        - fillText(텍스트,x,y[,maxWidth]): 지정된 위치(x,y) 기준 색이 채워진 텍스트를 그림.
        - measureText(텍스트): 현재 글꼴에서 주어진 텍스트 폭 반환(width속성값)
    - 관련 속성
        - font: 글자 스타일,크기,글자체 지정(기본값="10px sans-serif")
        - textAlign: 수평 방향 정렬방식 지정(값: left, right, center, <u>start</u>, end)
        - textBaseline: 텍스트 수직방향 기준선 지정(값: top, hanging, middle, <u>alphabetic</u>, ideographic, bottom)
2. 이미지 삽입하기: drawImage(이미지변수[,sx,sy,sw,sh],dx,dy[,dw,dh])

### 도형변환 -(5/5)
- 도형 변환: 캔버스의 좌표계 변환을 통해 그려지는 도형에 대한 변환효과 부여
    1. 이동 변환: translate(x,y)
    2. 크기 변환: scale(x,y) --1보다 크면 확대, [0,1]사이면 축소, 음수면 대칭(좌우/상하)
    3. 회전 변환: rotate(각도) --시계방향, 기준점=왼쪽상단모서리, 각도=라디안
    4. 사용자 정의 변환: transform(a,b,c,d,e,f)--이전에 호출한 값이 있다면 누적됨, setTransform(a,b,c,d,e,f)--누적X,항상 초기화됨
        - a,d: 수평/수직방향 확대축소
        - b,c: 수평/수직방향 기울임
        - e,f: 수평/수직방향 이동
        - `rotate(각도)`==`transform(Math.cos(각도), Math.sin(각도), -Math.sin(각도), Math.cos(각도), 0, 0)`

<!-- *** -->
## Application
### example review
### confusing point
### extension point
(implement)
## Reference(bookmark)


<!-- ------------------------------ -->
# Chapter 7. HTML API
>오디오,비디오파일제어/드래그앤드롭\
/웹스토리지/위치정보
### Outline
### Flow Diagram
<!-- *** -->
## Main Concepts
### 오디오/비디오 파일 제어 -(1/4)
1. js를 통해 미디어 파일 재생 제어 가능
    - 미디어 파일의 재생관련메서드: load(), play(), pause(), canPlayType()--특정 미디어파일 형식 재생가능여부 반환(값:probably, maybe,"")
2. 미디어 파일 제어/상태 관련 속성
    - 지정/반환
        - autoplay(로드시 자동재생여부), controls(플레이어표시), defaultMuted(기본음소거상태), defaultPlaybackRate(기본재생속도,기본값=1.0),
        - currentTime(현재재생위치-초단위),
        - loop(반복재생여부), muted(음소거여부), playbackRate(현재재생속도), volume(볼륨)
        - preload(이미지로드시 파일로드여부), src(미디어요소의 현재파일경로)
    - 반환
        - currentSrc(재생중 미디어파일url), error(미디어파일오류상태-`error.code`값),
        - duration(총 재생시간-초단위),
        - ended(재생종료여부), paused(일시정지여부)
3. 미디어 관련 주요이벤트
    - abort(로딩중지), error(로드/재생과정에 오류발생), loadstart(로드시작), waiting(다음프레임데이터 버퍼링으로 미디어멈춤), progress(미디어다운로드시 지속적발생)
    - canplay(재생시작가능), ended(재생종료시), pause(일시정지), play(재생시작), playing(일시정지/버퍼링을 위해 멈췄다가 재생시작)
    - seeked(유저가 미디어파일의 새로운위치로 이동/건너뜀을 마침), seeking("을 시작), timeupdate(현재 재생위치변화-재생중 지속적발생) 
    - durationchange(duration속성변화), ratechange(재생속도관련속성변화), volumechange(volume/muted속성변화)

### 드래그 앤 드롭 -(2/4)
- 드래그 앤 드롭: 마우스 이용, 특정 콘텐츠를 끌어다놓은 것
1. 드래그 속성 지정(`draggable="속성값"`): draggable속성 (-> true | false) --어떤 객체도 드래그 가능
2. 드래그 관련 이벤트 처리(`onXXX="메서드명(event)"`)
    - 드래그 속성 지정 시 이벤트 지정/컨트롤이 필요
    - 이벤트 종류와 발생 지점
        - 드래그소스: (dragleave) **dragstart**
        - 드래그과정: (dragenter, drag, dragleave), **dragover**
        - 드롭대상: (dragenter, dragend) **drop**
    - 이벤트에 따라 처리해야할 작업
        1. dragstart이벤트: 전달 데이터 저장 -> `setData()`메서드
        2. dragover이벤트: 기본값 취소해 요소위 다른요소가 드롭될 수 있도록 지정 -> `preventDefault()`메서드
        3. drop이벤트: 전달하려는 데이터를 가져와 드롭요소에 추가 -> `getData()`+`appendChild()`메서드
3. 전달 데이터 처리: dataTransfer객체(`event.dataTransfer`)
    - 드래그앱드롭으로 전달되는 데이터 처리, 이벤트가 발생하는 동안에만 유효하게 사용
    - 메서드
        - `setData(type,data)`: 드래그시작시 전달할 데이터 보관(type:"text","url" / data: 드래그되는요소id)
        - `getData(type)`: 드롭할때 setData()에서 저장한 데이터를 가져옴
        - `clearData([type])`: 듣래그앤드롭 도중 특정 포맷 데이터 삭제
        - `setDragImage(이미지변수,x,y)`: 드래그 도중 커서위에 표시할 사용자이미지 지정
4. 드래그 포인터 변경
    - effectAllowed속성(dragstar이벤트--드래그하는동안)
    - dropEffect속성(dragenter/dragover이벤트--드래그 끝날때)
    - 값: none, copy, link, move
### 웹 스토리지 -(3/4)
1. 웹 스토리지와 종류
    - 클라이언트측의 데이터 저장영역
    - vs 쿠키: 용량(도메인당5mb vs 4kb), 네트워크전송(http메세지에 미포함 vs http헤더에 포함,부하o,보안취약), 유효기간(로컬스토리지는 유효기간x vs 유효기간o), 세션(세션 스토리지는 각 윈도우마다 독립적 저장 vs 독립적 데이터저장불가<-모든 윈도우가 고유함)
    - 저장 데이터: `(키,값)`
        - 키를 통해서만 값 접근o, 중복 키x
        - 키,값 모두 대소문자 구분, 따옴표를 사용하는 문자열
    - 분류
        - 기준: (저장데이터의) 공유범위, 유효기간
        - localStorage: 데이터 저장기간 제한X, 도메인마다 별도의 저장영역(같은 도메인의 윈도우끼리는 접근가능)
        - sessionStrage: 데이터 저장기간 제한O(=세션 종료시), 세션마다 별도의 저장영역(같은 도메인; 다른 윈도우면 접근불가)
    - 웹 페이지 로드 시 클라이언트에 자동생성
    - 지원여부: `typeof(Strage)!=="undefined")` or `window.localStorage` or `window.seesionStorage`
2. 스토리지 객체 속성,메서드
    - 속성: `length`(저장 데이터 개수 반환)
    - 메서드: `key(index)`, `getItem(key)`, `setItem(key, value)`, `removeItem(key)`, `clear()`(저장된 모든데이터 삭제)
        - 데이터 저장: `localStorage.setItem(key, value);`, `localStorage.key=value;`, `localStorage[key]=value;`
        - 데이터 읽기: `변수=localStorage.getItem(key);`, `변수=localStorage.key;`, `변수=localStorage[key];`
        - 데이터 삭제: `localStorage.removeItem(key);`, `delete localStorage.key;`, `delete localStorage[key];`
3. 웹 스토리지 이벤트
    - window 객체의 `storage` 이벤트
        - 웹 스토리지에서 삽입,삭제,값 변경 등 변화발생시 다른 모든 윈도우에 변경사실을 전달
        - not세션스토리지
    - 속성
        - key(변화발생한 데이터의 키)
        - oldValue, newValue
        - url(이벤트가 발생한 웹문서url), storageArea(이벤트가 발생한 웹스토리지 객체)
 
### 위치 정보 -(4/4)
1. HTML geolocation API
    - 유저의 지리적 위치정보를 얻을때 사용(위치정보->개인정보->사용자동의 필요)
    - `window.navigator.geolocation` 객체 사용
        - 메서드: getCurrentPosition()-1회, watchPosition()-연속, clearWatch()
        - 지원여부 확인: `if(navigator.geolocation) { 위치정보획득 } else{ 위치정보미지원 }`
2. 현재 위치정보: `getCurrentPosition(successCallback, errorCallback, options)` 메서드
    - 현재 위치정보를 1번만 얻음
    - 파라미터
        - successCallback: 위치정보를 성공적으로 얻었을때 호출될 콜백함수명
            - 콜백함수의 인자로 전달: Position 타입 객체(위치관련정보)
                - 속성: `coords.latitude`(위도), `coords.ongitude`(경도), `coords.altitude`(고도-미터단위), `coords.accuracy`(위도,경도의 정확도-미터단위), `coords.altitudeAccuracy`(고도의 정확도-미터단위), `coords.heading`(이동중의 진행방향-북쪽기준 시계방향 각도), `coords.speed`(이동중 속도-초당 미터), `timestamp`(위치가 파악된 시간-밀리초단위)
        - errorCallback(생략가능): 오류발생시 호출될 콜백함수명
            - 오류콜백함수의 인자로 전달: error객체
                - 속성: `code`(현재 오류 코드 반환--`PERMISSION_DENINED`, `POSITION_UNAVAILABLE`, `TIMEOUT`, `UNKNOWN_ERROR`), `message`(오류 내용 텍스트)
        - options(생략가능): 선택사항, 객체형식 전달
            - enableHighAccuracy(기본값:true): 정확도 높은 위치정보 요청
            - timeout: 위치정보파악에 대한 제한시간설정, 밀리초단위(지나면 TIMEOUT오류발생)
            - maximumAge: 파악된 위치정보 유효시간, 밀리초단위
                - 초과 시 정보폐기후 새 위치정보 파악시도
3. 연속적인 위치정보: `watchPosition(successCallback, errorCallback, options)` 메서드
    - 사용자 이동에 따라 변하게 되는 현재위치정보를 연속적 추적
    - 파라미터: getCurrentPosition()과 동일
    - 반환: 식별ID반환
    - `clearWatch(식별ID)`: 연속적 위치추적을 해제시키는 메서드

<!-- *** -->
## Application
### example review
### confusing point
### extension point
(implement)
## Reference(bookmark)