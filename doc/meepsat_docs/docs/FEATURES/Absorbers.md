---
# Absorbers
---

[TOC]

## Introduction

Broadband Microwave Absorbers in CMB Telescopes helps in controlling stray light at millimeter wavelengths that requires special optical design and characterisation. In this section, we present the concepts and workflow used for creating different types of absorbers that can be used for various types of simulations (e.g., full telescope and unit cells simulations).

The concepts used here are mainly adopted from this [paper](https://arxiv.org/pdf/2511.05309). In the current version of MeepSAT, the following types of absorbers are supported:

- Pyramidal 
- Linear
- Exponential

Very soon, we will also implement other types of absorbers (Klopfenstein, Stepped) as well.


## Workflow

As described in this [paper](https://arxiv.org/pdf/2511.05309), we follow the following chain to define the different types of absorbers: 

$$\text{z(l)} \implies \epsilon\text{(l)} \implies \alpha\text{(l)} \implies \text{Required Geometry}$$

- $\text{z(l)}$ is the impedance profile of a particular taper
- $\epsilon\text{(l)}$ is the relative permittivity at l
- $\alpha\text{(l)}$ is the filling factor at l
- l varies from location 0 to h where h is the height of the taper.


Assuming p to be the base, h be the height of the absorber and $\theta$ be the angle of the absorber w.r.t normal axis. At l = 0, the impedance matches the impedance of the lossy dielectric, $Z_1 ≡ Z_0/\epsilon_r$ and $Z(h) = Z_0$ representing the impedance of the medium just outside the tip of the taper.

**Step 1: Choose the type of taper**

- Pyramidal

$$\alpha(l) = \left(\frac{p - 2\tan(\theta)l}{p}\right)^2$$

- Linear

$$Z_{\text{lin}}(l) = (\frac{Z_1 - Z_0}{h})l + Z_0$$

- Exponential

$$Z_{\text{exp}}(l) = Z_1 e^{\frac{l}{h}\ln{\frac{Z_0}{Z_1}}}$$

**Step 2: Calculate the relative permittivity at location l**

$$\epsilon(l) = (\frac{Z_0}{Z_l})^2$$

**Step 3: Calculate the filling factor $\alpha(l)$**

$$\alpha(l) = \sqrt{\frac{\epsilon(l) - 1}{\epsilon_r-1}}$$

**Final Step: Extracting the geometry**

The geometry of the absorber can then be constructed along the height h via

$$p^2 \alpha(l)$$

## Implementation in MeepSAT