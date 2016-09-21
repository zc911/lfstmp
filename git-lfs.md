## Git lfs 使用方法

```
Matrix使用git lfs对大文件进行版本管理。目前只管理了 Matrix 的依赖库，后续会根据情况把模型文件也加入进来。使用方式如下：
```

1.安装git lfs client

```shell
$ wget http://192.168.2.119/tools/git-lfs-linux-amd64-1.4.1.tar.gz
$ tar -xzvf git-lfs-linux-amd64-1.4.1.tar.gz && cd git-lfs-1.4.1
$ sudo ./install.sh
$ git lfs version
# 查看安装是否成功
```

2.配置lfs server

```shell
$ git config --global --add lfs.url "http://matrix:matrix@192.168.2.119:9999"
$ git config -l
# 查看配置是否生效
```

3.开始使用

```shell
$ cd MATRIX_DIR
$ git lfs install
$ git pull
# 如果要求lfs的用户名和密码，均为matrix
```

4.常用操作

```shell
$ git lfs track "lib/*/*/*"
# 指示lfs跟踪lib文件夹及子文件夹下的所有大文件
$ git add .
$ git commit -m "lfs track"
$ git push
# 与普通的git使用方式一致，把大文件add和commit到工作区后git push将更改更新到server。其中代码更新到github，大文件更新到本地的lfs server
# git push 如果要求lfs的用户名和密码，均为matrix

$ git lfs status
# 查看当前lfs的跟踪状态

$ git lfs help
# 查看帮助
```
