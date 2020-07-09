# [Go] Hugo Quick Start


[Original Source](https://gohugo.io/getting-started/quick-start/) 

# Hugo 설치

---

## Ubuntu

```bash
sudo apt-get install hugo 
```

## Windows

릴리즈 버전만 [깃허브](https://github.com/gohugoio/hugo/releases)에서 다운 받아서 `C:\Hugo\bin`에 넣어주고 환경변수 셋팅만 해주면 된다.

# Hugo `new`

---

new 키워드는 새로운 프로젝트를 만들때 사용할 수 있고,

```bash
hugo new site quickstart
```

새로운 포스트를 만들때도 사용할 수 있다.

```powershell
hugo new posts/aa
```

# Hugo `theme`

---

```bash

cd quickstart
git init
git submodule add https://github.com/budparr/gohugo-theme-ananke.git themes/ananke
git submodule add https://github.com/dillonzq/LoveIt.git themes/loveit
```

- https://github.com/dillonzq/LoveIt.git

theme를 가져왔는데 hugo server -D 가 안먹힌다? 

→ extended version을 사용해보자

# Hugo `run`

---

```bash
hugo server -D
```

이제 우리는
