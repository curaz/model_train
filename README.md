# BOAZ 21기 꾸라즈 ADV 프로젝트: 인생책을 찾아주는 큐레이션 추천시스템

> [BOAZ 발표자료 바로가기](https://www.slideshare.net/slideshow/20-boaz-pdf/270919489)


## 목표/문제
<div align="center">
  <img src="https://github.com/user-attachments/assets/d27770f0-8861-4c00-b6df-5a53128cc86f" width="600" />
</div><br/>

- 유저의 explicit data(좋아요, 읽은 책, 인생책 정보)를 활용하여 유저가 선호할만한 도서를 추천하는 것이 목표입니다.
- 추천 방법은 총 2가지인데, 메타데이터를 기반으로 팔로우할 만한 유저 추천, 같은 아이템에 관심을 가진 유저가 선호한 다른 아이템을 추천하도록 합니다. 즉 1번 - 유사한 유저, 2번 - 선호할 만한 아이템을 출력하도록 모델을 학습합니다.
- 업로드된 유저 관련 csv는 모두 hash id로 처리되어있습니다.

## 모델 추론


### requirements
```
scikit-learn==1.3.0
scipy==1.11.1
numpy==1.23.5
pandas==2.0.3
torch==2.1.2
python == 3.11.5
tqdm==4.65.0
```


### Run python
실행시 top-K의 hash id의 결과를 받을 수 있습니다.
```bash
python main_kgat.py
```