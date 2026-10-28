.. currentmodule:: pypillometry.pupildata

PD-area change due to distortion caused by off-center gaze
===========================================================

We assume that the plane of the pupil is located parallel to the 
camera plane and when the eye is fixating the middle of the screen $(0,0)$, 
then the pupil is a perfect circle with radius $r$. In that case, the 
recorded pupil signal will be the area of a circle 

$$
A=\pi r^2.
$$

However, when fixating another point on the screen $(x,y)$, there is a 
displacement where $\alpha$ is the displacement angle in x-direction 
and $\beta$ in y-direction. Because the pupil is a circle, we can reparametrize $(x,y)$ to 
$(x',y')=(\sqrt{x^2+y^2},0)=(d,0)$ so that the displacement 
has only one dimension parametrized by angle $\theta$.

When $\theta>0$ (i.e., the eye is not perfectly fixating $(0,0)$), 
the pupil plane is tilted relative to the camera plane. 
When projecting the tilted pupil onto the camera plane, the radius in this 
direction is shortened to $a=r\cos \theta$. Therefore, the projection 
of the pupil is an ellipse with parameters $a$ and $r$ and the actual area of 
this ellipse is 
$$
A'=\pi a r=\pi\cos\theta r.
$$

When fixating point $(x,y)$, the distance of this point from the center 
is $d=\sqrt{ x^2+y^2}$. When the eye is a distance of $h$ away from the 
screen, then the displacement angle is

$$\theta=\arctan\frac{d}{h}$$

such that

$$a=r\cos\arctan\frac{d}{h}$$

which `wolframalpha` assures us is equal to

$$a=\frac{r}{\sqrt{1+\frac{d^2}{h^2}}}$$

and therefore 

$$A=\frac{\pi r^2}{\sqrt{1+\frac{d^2}{h^2}}}$$
  
Therefore, we can correct the raw signal $ which we assume is the area of a 
circle with radius $r$. 
