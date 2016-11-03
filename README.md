# CarlosFuerte
Charlie Strong's Turtleneck


<img src="https://s3.amazonaws.com/cuttings/backgrounds/291230/Charlie%20Strong.jpg" width="300px" style="display: inline-block; margin-top: 5px;">

Links:

http://rogerdudler.github.io/git-guide

# Forking a Repository
https://help.github.com/articles/fork-a-repo/

## Step 1:
Navigate to https://github.com/davidgtang/CarlosFuerte and hit the fork button at the top right.

## Step 2:
Clone your fork in location of your choice
```
git clone YourForkAddress
```

## Step 3:
Change directory into your local fork
```
git remote -v
```

You should see the your local fork for (push) and (fetch)

Now add the upstream remote:
```
git remote add upstream https://github.com/davidgtang/CarlosFuerte
```

# Syncing Master to Local Fork
https://help.github.com/articles/syncing-a-fork/
Everytime you start working, make sure you're synced to the upstream master.
```
git fetch upstream
git checkout master
git merge upstream/master
```

