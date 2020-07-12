---
title: "[Hugo] Quick Start 1"
date: 2020-07-10T01:34:34+09:00
draft: false 
tags: ["hugo", "quick-start"]
categories: ["Development"]

resources:
- name: "featured-image"
  src: "https://gohugo.io/images/gohugoio-card.png"
- name: "featured-image-preview"
  src: "https://gohugo.io/images/gohugoio-card.png"

description: "This article shows the basic Markdown syntax and format."

# cover: "https://gohugo.io/images/gohugoio-card.png"
lightgallery: true


---

<img src="https://gohugo.io/images/gohugoio-card.png" />

# Introduction

Nowadays, many people are blogging to provide good quality information.

In South Korea, various information has been provided through Daum and Naver Blog, and Tistory.

Recently, more high-quality information is provided worldwide through media such as github blog and medium.

Therefore, in this post, I will introduce the `Go` language based Static Website Generator **Hugo**, among the many techniques that help you create a github blog.

Through the Hugo Quick Start series, you can learn about OS-specific installation, key commands, blog preview and creation. 

Since git is required for convenient operation, so you need to configure the git environment.

- `git`: https://git-scm.com/downloads


## 1. Install Hugo 

As a first step to creating a blog with Hugo, of course, you need to install Hugo.

The environment I am working on is [Ubuntu 20.04](https://releases.ubuntu.com/20.04/), a Linux environment.

For Unix-based Linux and MacOS, installing Hugo is fairly straightforward.

### 1.1. Ubuntu

```bash
sudo apt-get install hugo 
```

### 1.2. MacOS

```bash
brew install hugo
```


### 1.3. Windows

For Windows, download only the release version from [Hugo](https://github.com/gohugoio/hugo/releases), put the extracted file under the C drive, and set environment variables.

For example, I put the files in `C:\Hugo\bin` and added the folder path to the environment variable.

If you set the environment variables like this, you can use the Hugo command by launching **cmd** or **powershell** in Windows.

<br /> 

If you have installed Hugo for your OS using the above methods, check the current Hugo Version with the following command to check if the installation was successful.

```bash
hugo version
```

![hugo version check](/images/hugo_quick_start/hugo_version.png)

## 2. Create Hugo Site

Now you can use a variety of features, including hugo's site creation feature.

### 2.1. `new site`

There are two things you can do with the `new` keyword, which can be used when creating a new blog project.

```bash
hugo new site quickstart
```

With the above command, we created a new blog project in the folder named `quickstart`.
 
The project has a seemingly complex structure, but I will explain this structure as detail in the following post.

### 2.2. `new posts`

The `new` keyword can also be used to create new posts.

```bash
cd ./quickstart
hugo new posts/my-first-post.md
```

Through this command, a [markdown](https://gist.github.com/ihoneymon/652be052a0727ad59601) document was created in our project called `content/posts/my-frist-post.md`.

## 3. Use Hugo Theme

In the current state, nothing will be seen even if you run the project.

This is because there is currently no theme specified for how to show the blog.

You can also develop and use themes directly in the `themes` folder, but it is also a good idea to quickly apply well-made themes.

> Hugo Theme List: https://themes.gohugo.io/

### 3.1. Download theme 

For first-time users, we will apply a simple yet beautiful [Ananke](https://themes.gohugo.io/gohugo-theme-ananke/) theme.

After initializing git in the folder as follows, you should add the theme you want to use as `submodule` rather than receiving it as `clone`.

```bash
git init
git submodule add https://github.com/budparr/gohugo-theme-ananke.git themes/ananke
```

The `themes/ananke` at the end means that the theme will be downloaded and saved under the `ananke` folder which is in the `themes` folder of project.

### 3.2. Apply theme using `config.toml`

If the theme has been downloaded, you should modify the `config.toml` in the project folder so that the theme is applied.

To the contents of the initial file as follows: 

```toml
baseURL = "http://example.org/"
languageCode = "en-us"
title = "My New Hugo Site"
```

Just add a line to `theme = ananke` (the folder name created earlier).


## 4. Run and Generate Hugo

### 4.1. Preview blog

Now you can finally see how your blog will look.

```bash
hugo server -D
```

When you run that code, you can see the theme applied to the blog and the posts that have been made.

![hugo server -D](/images/hugo_quick_start/hugo_server.png)

Originally, the default port should be *1313*, such as `localhost:1313` or `127.0.0.1:1313`, but when running multiple ports, the remaining ports are randomly specified.

### 4.2. Generate blog

The previous command just provided a preview of how the blog will look on your local environment, however, the following command creates a real website consisting of `html`, `css`, `javascript`, etc.

```bash
hugo -D
```

A new public folder is created, and the files are generated accordingly. When you upload the generated file to the github repository to host, the blog is hosted.

# Conclusion

In this post, we learned about the process from installing Hugo to generating blog.

In the next **Hugo Quick Start** post, I'll cover the details of the Hugo structure, starting with how to host a generated website on Github.

Thank you.

# Reference

- [Quick Start | Hugo](https://gohugo.io/getting-started/quick-start/) 



