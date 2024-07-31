# Transformation Classes 

Each functional transform has its `class` counterpart, except for [polar.train.data.transforms.functionals.pairwise_max_norm][].        
In almost all cases, the idea is to have a determinist functional implementation and a random class counterpart.

## Common preprocessing

::: polar.train.data.transforms.transforms.Center

::: polar.train.data.transforms.transforms.Normalize

::: polar.train.data.transforms.transforms.RandomSample

---

## SIM(3)

::: polar.train.data.transforms.transforms.RandomTranslate

::: polar.train.data.transforms.transforms.RandomRotate

::: polar.train.data.transforms.transforms.RandomRigidMotion

::: polar.train.data.transforms.transforms.RandomScale

---

## Augment


::: polar.train.data.transforms.transforms.RandomJit

::: polar.train.data.transforms.transforms.RandomPlaneCut