# Matrix Engine
### Version 0.2.0
2016-07-13

```
IMPORTANT: The dg plate sdk found multi-thread bug. So only one thread support in engine
```

### Feature
- Update libcaffe, to fix SSD Detection multiple thread bug. Now must use cudnn v4
- Update dg plate lib to 2.7.1.1 to try to fix crash bug and 119M GPU0 bug
- Use mulitiple thread when processing plate
- Remove pedestrian assert
- Add feature check when init witness engine


