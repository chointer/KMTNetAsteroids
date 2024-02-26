### KMTNet 다중 필터 소행성 측광 
외계행성 탐색시스템 다중 필터 소행성 관측 자료로부터 밝기를 측정하는 코드입니다.  
4개의 다중 필터(g, r, i, z 혹은 B, V, R, I)로 관측된 자료를 처리해 총 9,000여 개의 소행성을 측광했습니다.
<br/><br/>

- **연구 기간**  
  2018.09. - 2022.05.
<br/><br/>

- **기술 스택**  
  Python, shell script
<br/><br/>

- **Background**
<br/><br/>

- **자료 처리 과정**
<br/><br/>

- **구조**
```
KMTNetAsteroids/
├── README.md
├── MyModule/                      # 재사용 가능한 기능을 모듈로 정리
│   ├── ModuleCommon.py            ## deg 단위 변환 함수, 오차 전파 계산 함수, 선형 회귀 with sigma-clipping
│   ├── ModulePCA.py
│   ├── ModuleRefDown.py
│   ├── PanSTARRS.py
│   └── standardization.py
├── analysis/                      # 측광 결과 분석
│   └── analysis_error.py
└── photometry/                    # 측광 코드
    ├── KMTNet_0_RefDownload.py    ## 관측 하늘에 위치한 천체 정보 다운로드 함수
    ├── KMTNet_1_match.py          ## 광원과 별, 소행성 동정 함수
    ├── KMTNet_2_stdzation.py      ## 등급 표준화 함수
    ├── KMTNet_3_gather.py         ## 중복 관측 결과 통합 함수
    └── KMTNet_TotalRun.py         ## 전체 측광 코드 실행
```
