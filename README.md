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
├── MyModule/
│   ├── ModuleCommon.py
│   ├── ModulePCA.py
│   ├── ModuleRefDown.py
│   ├── PanSTARRS.py
│   └── standardization.py
├── analysis/
│   └── analysis_error.py
└── photometry/
     ├── KMTNet_0_RefDownload.py
     ├── KMTNet_1_match.py
     ├── KMTNet_2_stdzation.py
     ├── KMTNet_3_gather.py
     └── KMTNet_TotalRun.py
```
