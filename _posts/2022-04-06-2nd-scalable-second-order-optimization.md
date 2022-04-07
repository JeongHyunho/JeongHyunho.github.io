---
layout: post
title: "[정리] Scalable Second Oder Optimization for Deep Learning (2021)"
---

Author: Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, Yoram Singer
Paper Link: https://arxiv.org/abs/2002.09018

## 요약


## Abstract
* 1차 경사 하강법에 비해 2차 최적화 방법은 계산량, 통신량이 많고 메모리 소모가 커 잘 사용되지 않았음
* 큰 스케일 문제에 사용될 수 있는 2차 preconditioned method 를 제안함
* 높은 수렴성과 1차 경사 하강법 대비 wall-clock 기준 빠른 업데이트를 확인함
* 여러 CPU 와 가속기(?)를 활용한 결과로, BERT, ImageNet 같은 큰 학습 문제에서 성능 확인함

## Introduction
* 2차 방법에서 계산량과 메모리 소모를 줄이는 것이 중요하며, 학습에 필요한 스텝 수를 줄일 수 있음
* 확률 최적화를 위한 *적응형* 방법을 제안하며, [AdaGrad](https://jmlr.org/papers/v12/duchi11a.html), [Adam](https://arxiv.org/abs/1412.6980) 의 full matrix 버전이라 할 수 있고 parameter 사이 correlation 을 고려하게 됨 (이 방법들은 gradients 의 외적합을 이용하여 2차 모멘트 행렬을  계산했음)
* 2차 방법의 문제점들을 완화하기 위해 [K-FAC](https://arxiv.org/abs/1503.05671), [K-BFGS](https://arxiv.org/abs/2006.08877), [Shampoo](https://arxiv.org/abs/1802.09568) 가 제안되었지만, 큰 스케일의 딥러닝을 위해 병렬화하기 어려움

### Contribution
* 많은 레이어들을 갖는 딥러닝 모델을 학습하기 위해 Shampoo 방법을 확장하였음
* Preconditioner 행렬을 위해 계산량이 많은 SVD 을 PSD 행렬을 위한 안정적인 방법으로 대체함
* Machine translation, Language modeling 등에서 학습 속도 (Wall-clock) 향상을 확인하였음

### Related Work
* [Bollapragada et al., 2018](https://arxiv.org/abs/1802.05374) 은 노이즈가 적은 Full batch L-BFGS 와 확률 경사 사이의 방법을 제안함
* 대부분의 이전 Preconditioner 방법은 대각 행렬 근사를 이용하였으나, 최근에 들어 full matrix 를 이용하는 방법들이 제안되었음
	* K-FAC 은 Fisher 행렬을 근사하여 preconditioner 로 사용하는 것이며, K-BFGS 는 레이어의 Hessian 을 유사한 방법으로 근사함
	* [GGT](https://arxiv.org/abs/1806.02958) 는 AdaGrad preconditioner 를 row-rank 근사하는 방식이나 많은 수의 gradients 를 저장해야므로 중간 정도 크기의 모델에 적합함
* [Ba et al., 2017](https://jimmylba.github.io/papers/nsync.pdf) 은 K-FAC 의 distributed 버전을 제안했음

## Preliminaries

### Notation
* Frobenius norm of A: $||A||^2_F=\sum_{i,j}A^2_{i,j}$
* Element-wise product of A and B: $C=A\bullet B\iff C_{i,j}=A_{i,j}B_{i,j}$
* Element-wise power: $(D^{\odot\alpha})_{i,j}=D^{\alpha}_{i,j}$
* $A\preceq B$ iff $B-A$ is positive semi-definite (PSD)
* For a symmentric PSD matrix $A=UDU^\text{T}$, $A^\alpha=UD^\alpha U^\text{T}$
* Identities of the Kronecker product of A and B:
	* $(A\otimes B)^\alpha=A^\alpha\otimes B^\alpha$
	* $(A\otimes B)\mathbf{vec}(C)=\mathbf{vec}(ACB^\text{T})$

### Adaptive preconditioning methods
* Parameter $w_t\in\mathbb{R}^d$ 는 gradient $\bar{g}_t\in\mathbb{R}^d$ 와 precondition 행렬 $P_t\in\mathbb{R}^{d\times d}$ 에 대해 다음과 같이 업데이트됨

$$
w_{t+1}=w_t-P_t\bar{g}_t
$$

* $P_t$ 는 2차 최적화 방법에서 Hessian 행렬이 되지만, adaptive preconditioning 방법에서는 gradients 의 correlation 과 관련됨
* Parameter 와 gradient 를 행렬 $W, G \in\mathbb{R}^{m\times n}$ 로 각각 표현 했을 때, full matrix 일 경우의 precondition 의 저장공간은 $n^2m^2$ 만큼 필요하며, 업데이트 식의 계산량은 $m^3n^3$ 이므로 근사 없이는 계산이 불가능함

### The Shampoo algorithm
* Shampoo 는 Kronecker product 를 이용하여 대각행렬 근사 방법과 full matrix 방법을 잇는 방법임
* Iteration $t$ 의 손실 함수에 대한 gradient $G_t=\nabla_Wl(f(W,x_t),y_t)$ 에 대해 $L_t\in\mathbb{R}^{m\times m}$ 와 $R_t\in\mathbb{R}^{n\times n}$ 는 아래같이 정의됨
$$
L_t=\epsilon I_m+\textstyle\sum^t_{s=1}G_sG_s^{\text{T}} \quad
R_t=\epsilon I_n+\textstyle\sum^t_{s=1}G_s^{\text{T}}G_s
$$
* Full matrix Adagrad preconditioner $H_t$ 는 $(L_t\otimes R_t)^{1/2}$ 로 근사되며, Shampoo 는 아래 업데이트 식을 따름
$$
W_{t+1}=W_t-\eta L_t^{-1/4}G_tR_t^{-1/4}
$$


## Concluding Remarks
* 딥러닝을 위한 2차 최적화 방법을 구현 방법을 제안하였고 성능을 확인함
* 기존 구현 대부분이 대칭 행렬을 이용하지만, 대칭 연산자를 이용하는 경우는 발견하지 못했는데, 이는 플롭과 메모리를 약 50 % 절약할 수 있음
* 1차 방법에 맞춰진 여러 최적화 방법은 2차 방법에도 적용될 수 있음
	* 예를 들어 ... XX
* LAPACK 과 같이 하드웨어 보조가 있으면

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEwODMzNTI4OTQsLTk2MjkzMDg3MSwtMT
AwMTg0NzkyNiwxNDMzMTc3MTgzLC0xNzk2NTkzODE0LC0xMjYw
MjI1MjA5LC0xMTczNzg4MDU1LC0xMTQ5OTkyMzUwLDEyMTc4ND
A3MzYsLTgxMjY5MDI4OCwyMTU1NjA1NzMsLTE2MzU4NDA3NzQs
LTkyNzk1NTM2OV19
-->