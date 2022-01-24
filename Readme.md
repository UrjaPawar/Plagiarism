---
 layout: post
 title: Evaluation of Feature-based Explanations
 tags: [Explainable AI, Evaluating XAI , Robustness Analysis]
 authors: Anonymous
 <!-- authors: Pawar, Urja, Munster Technological University, Ireland; -->
 ---

<!-- content -->
### Introduction
The rising need of making Machine Learning (ML) models interpretable, fair and trustworthy has led the research community to come up with better explanations to enable interpretation, validation, and transparency while utilising these models in domains such as healthcare or finance. In case you aren’t familiar with the domain of explainability in AI, explainable AI comprises of set of all techniques that are developed to explain predictions or classifications made by AI-based systems. 
But, how do we judge whether the explanation is better or not? Different types of explanations require different evaluation metrics to get assessed. The most common type of explanation is feature importance explanations that are usually presented in the form of relative ranking of feature as per their importance in determining model’s output. Another common type is counterfactual explanations that tells us what minimum changes are required in the features that will result in different classification by the model. 

#### Feature-based Explanations
Both feature importance and counterfactuals present explanations based on features. While feature importance scores the impact of different features, counterfactuals highlight the features to change (and the degree/nature of change) in order to achieve a desired classification. These explanations can, therefore, be called feature-based explanations.
#### Robustness
For black-box classifiers, feature importance and counterfactuals are estimated using different techniques that involves either training an interpretable classifier to mimic black-box in local neighbourhood, or perturbations based on insertion and removal of features by capturing how the insertion/removal impacts model's output. As these are based on estimation or approximation techniques, a set of evaluation metrics are required to represent the faithfulness of explanations to assess whether or not they truly represent the black-box model. These metrics represent the “robustness” of explanations.

### What’s this about?
This blog post describes contribution of the paper titled "Evaluations and Methods for Explanations Through Robustness Analysis" by [Cheng et. al](https://openreview.net/forum?id=Hye4KeSYDr) that discusses assessing the robustness of an explanation in a novel way and presents a more robust explanation. The contribution lies in the specific domain of Explainable AI (XAI) methods that generate explanations by capturing the impact of insertion and removal of features on model’s output. Explainable AI methods that are based on insertion/removal face two drawbacks. Let’s assume that we want to estimate feature importance from such XAI techniques, then:
1.	 When the feature importance is estimated by removing a feature by setting it to a baseline value, it has higher chance to attribute high importance to a feature if some values deflect a lot from baseline. An example would be setting RGB pixels to black, that will give bright pixels more importance.
2.	When the feature importance is estimated by removing a feature by giving it some value sampled from the distribution (using a generative model), there is an inherent bias that goes from the generative model to this process, this bias deflects the feature importance from representing a truer picture. Additionally, not all domains can have a proper generative models.


To address these drawbacks, the paper contributes towards:
1.	Proposing novel evaluation criteria to assess the robustness of feature based explanations -  feature importance and counterfactuals - based on small perturbations to understand how the model's output changes with these small perturbations on different set of features.
2.	Better explanations by optimising them based on the novel evaluation criteria 

#### Assumptions:
In order to have an evaluation of the explanations, two key things are assumed in the paper:
1. Changing the values of only *non-important* features will have a weak influence on model's output
2. Changing the values of only *important* features can easily change model's output


Now based on the above set of assumptions, a robustness parameter ε<sup>*</sup> is defined as per the following equation:
$$\begin{equation}
ε^*_\mathrm{**x**s}= *g(f,**x**, S)* = min<sub>δ</sub>|δ| s.t. f(x+δ) \neq y, δ<sub>S<sup>c</sup></sub> = 0
\end{equation}$$

In the above equation, *f* is the model, *x* is an input vector, *y* is the original output of *x*, *U* is the set of all features and *S* is the subset of *U* and *S<sup>c</sup>* is the complementary subset of *S (U-S)*. The term *δ* is the minimum adversarial perturbation that is performed on subset *S* such that feature values in *S* changes by * δ* and we get a different output than *y*. In case of binary classification, it could be the opposite class. In case of multi-class classification, it could be any other class than *y*.
This minimum perturbation, when done on, let’s say, the set of less important features: **S<sup>c</sup>**, should be **high** as per assumption 1. This is because, if the features are less important: they should require heavy changes to be made to them for causing a change in model’s output.
Similarly, minimising *δ* over a set of important features **S** should give relatively **low** *δ* value, as only little perturbations are required to change model's output if we are perturbing important features. The robustness evaluation metric R can be defined as ε<sup>*</sup><sub>**x**s</sub> where S is any subset of features. Now, for convenience, R(S) and R(S<sup>c</sup>) are defined below.

> Based on this, we can say: R(S) - where S is a set of important features - is given by ε<sup>*</sup><sub>**x**s</sub> and,

> R(S<sup>c</sup>); where S<sup>c</sup> is a set of non-important features - is given by ε<sup>*</sup><sub>**x**s<sup>c</sup></sub>.

#### AUC (Area-Under-Curve) Analysis
With this evaluation metric R, we can also have a look at the AUC curve that is plotted against the top K features belonging to that subset. The top K features essentially ranks the top features in decreasing order of their R score while optimising the equation discussed previously.

![AUC]({{ site.url }}/public/images/2021-12-01-Evaluation of Feature-based Explanations/nimp.png) 

As shown in the AUC plot A, we should expect a large AUC if we use the R(S<sup>c</sup>) metric because of large R caused by requirement of large *δ*. The least important feature (K=1) shows the highest R(S<sup>c</sup>) value and subsequently as we progress towards less *non-important* features (or more important features), we get a lower R(S<sup>c</sup>) values.

![AUC]({{ site.url }}/public/images/2021-12-01-Evaluation of Feature-based Explanations/imp.png) 
Similarly, in plot B, low AUC is obtained on using the R(S) metric. The first feature being highly impactful shows the least R(S) values and as we move towards less important features, the area starts to increase. 

### Counterfactual flavor
If we optimise the previous equation of ε<sup>*</sup><sub>**x**s</sub> to the following equation:
$$\begin{equation}
ε^*_\mathrm{**x**s}= *g(f,**x**, S)* = min<sub>δ</sub>|δ| s.t. f(x+δ) = t, δ<sub>S<sup>c</sup></sub> = 0
\end{equation}$$

We can see that if the optimisation function can optimise for perturbations that lead to another desired class *t*, and in this way, it can provide us with the counterfactual use of the S subset of features. If we get the features that show a high ε<sup>*</sup><sub>**x**s</sub> value and capture the perturbation amount in each of them, we can present counterfactuals. As an example, in bird classification, if only changing the color of pixels in the eye region (important set of pixels) by little amount can lead to an image being classified as an altogether different bird, we can present the changed or manipulated image as the counterfactual explanation. 
### Extracting explanations

Based on *g(f,**x**, S)*, we can extract a set of important and non-important features by solving the following set of optimsation problems respectively, which are essentially optimisations over R(S) and R(S<sup>c</sup>) metrics.
$$\begin{equation}
minimise *g(f,**x**, S)* s.t. |S| <= K \\
maximise *g(f,**x**, S<sup>c</sup>)* s.t. |S<sup>c</sup>| <= K `
\end{equation}$$
where K is the top K number of features we intend to analyse or consider.

As described in the paper, the above equations could be solved by a greedy approach where we initialise an empty set S (or S<sup>c</sup>) and keep on adding features that most optimises the corresponding optimisation function. However, there is a drawback of missing the interaction among features. Two feature might be very important when put together and not important in a standalone manner. For addressing this, marginal contribution of a feature is also taken into consideration by analysing the change in model's output when some unchosen features are also included with this feature. This concept is based on game theory and can be used to optimally decide the contribution of feature on model's output by taking into consideration the feature interaction as well.

To avoid confusion, this evaluation criteria and explanations is different from SHAP in a way that SHAP considers removal of features by setting it to a baseline value, whereas here we are more interested in capturing change in model's output by slightly changing the input space from their original value. This change can then be used for optimisation to have more deflection in output (for important features) or less deflection (for non-important features).
