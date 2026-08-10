# References

Every entry below was checked against the published record before it was cited
anywhere in this package: DOI resolved, and authors, year, venue and the
specific claim confirmed against the publisher page, PubMed or Crossref. Four
entries had metadata corrected during that pass. Nothing here was cited on the
strength of looking plausible.

Annotations say what *this package* uses the reference for, which is not always
what the paper is best known for. Where a derivation in [theory.md](theory.md)
is this package's own rather than a cited result, that is said at the point of
use rather than implied by an adjacent citation.

## Multi-trait and multi-score prediction

Where the `w = C^-1 rho` combination comes from, and the alternatives to it.

- **Smith HF** (1936). A discriminant function for plant selection. *Annals of Eugenics 7(3), 240-250*. [doi:10.1111/j.1469-1809.1936.tb02143.x](https://doi.org/10.1111/j.1469-1809.1936.tb02143.x)
  The origin of the linear selection index: the w = C^-1 rho form that meta_pgs and wMT-SBLUP both instantiate, ninety years before either.
- **Hazel LN** (1943). The genetic basis for constructing selection indexes. *Genetics 28(6), 476-490*. [doi:10.1093/genetics/28.6.476](https://doi.org/10.1093/genetics/28.6.476)
  The genetic formulation of the selection index — index weights in terms of phenotypic and genetic covariance matrices, b = P^-1 G a — which is exactly the multi-trait BLUP weight formula in modern notation.
- **Bates JM, Granger CWJ** (1969). The combination of forecasts. *Journal of the Operational Research Society (Operational Research Quarterly) 20(4), 451-468*. [doi:10.1057/jors.1969.103](https://doi.org/10.1057/jors.1969.103)
  The non-genetics ancestor of meta_pgs: combining correlated predictors of one quantity with inverse-error-covariance weights, and the origin of the practitioner's preference for robust weights over exact Sigma^-1.
- **Jia Y, Jannink JL** (2012). Multiple-trait genomic selection methods increase genetic value prediction accuracy. *Genetics 192(4), 1513-1522*. [doi:10.1534/genetics.112.144246](https://doi.org/10.1534/genetics.112.144246)
  The cleanest empirical statement of the (1 - R_f^2)^2 factor derived in the theory notes: multi-trait gains concentrate on low-heritability targets with a well-powered correlated trait available.
- **Maier R et al.** (2015). Joint analysis of psychiatric disorders increases accuracy of risk prediction for schizophrenia, bipolar disorder, and major depressive disorder. *American Journal of Human Genetics 96(2), 283-294*. [doi:10.1016/j.ajhg.2014.12.006](https://doi.org/10.1016/j.ajhg.2014.12.006)
  The individual-level multi-trait GBLUP that wMT-SBLUP later approximated from summary statistics; establishes the effective-sample-size framing of multi-trait gains.
- **Hu Y, Lu Q, Liu W, Zhang Y, Li M, Zhao H** (2017). Joint modeling of genetically correlated diseases and functional annotations increases accuracy of polygenic risk prediction. *PLoS Genetics 13(6), e1006836*. [doi:10.1371/journal.pgen.1006836](https://doi.org/10.1371/journal.pgen.1006836)
  PleioPred: Bayesian family-(a) combination across genetically correlated diseases, and the closest pleiotropy-aware analogue of LDpred that multipgs's own scores could have been built with.
- **Márquez-Luna C, Loh PR, South Asian Type 2 Diabetes (SAT2D) Consortium, SIGMA Type 2 Diabetes Consortium, Price AL** (2017). Multiethnic polygenic risk scores improve risk prediction in diverse populations. *Genetic Epidemiology 41(8), 811-823*. [doi:10.1002/gepi.22083](https://doi.org/10.1002/gepi.22083)
  The minimal K=2 case of family (b): two scores, weights learned in the target population — the simplest demonstration that learned combination beats either input.
- **Krapohl E et al.** (2018). Multi-polygenic score approach to trait prediction. *Molecular Psychiatry 23, 1368-1374*. [doi:10.1038/mp.2017.163](https://doi.org/10.1038/mp.2017.163)
  The earliest clean statement of family (b) at scale in humans: regularized regression over dozens of PGS with no assumptions about the relationships among predictors.
- **Maier RM et al.** (2018). Improving genetic prediction by leveraging genetic correlations among human diseases and traits. *Nature Communications 9, 989*. [doi:10.1038/s41467-017-02769-6](https://doi.org/10.1038/s41467-017-02769-6)
  wMT-SBLUP / SMTpred: the direct ancestor of meta_pgs — a selection index over single-trait predictors with weights derived from h2, rG and N rather than learned.
- **Turley P, Walters RK, Maghzian O, Okbay A, Lee JJ, Fontana MA, et al. (Social Science Genetic Association Consortium)** (2018). Multi-trait analysis of genome-wide association summary statistics using MTAG. *Nature Genetics 50(2), 229-237*. [doi:10.1038/s41588-017-0009-4](https://doi.org/10.1038/s41588-017-0009-4)
  The canonical 'combine the summary statistics BEFORE scoring' alternative — variant-specific borrowing, in contrast to multi-PGS's one-scalar-per-trait weights.
- **Chung W et al.** (2019). Efficient cross-trait penalized regression increases prediction accuracy in large cohorts using secondary phenotypes. *Nature Communications 10, 569*. [doi:10.1038/s41467-019-08535-0](https://doi.org/10.1038/s41467-019-08535-0)
  CTPR: multi-trait prediction via a cross-trait penalty on per-variant effects rather than a cross-trait prior — another family-(a) design, and a direct competitor to MTAG for prediction.
- **Grotzinger AD et al.** (2019). Genomic structural equation modelling provides insights into the multivariate genetic architecture of complex traits. *Nature Human Behaviour 3(5), 513-525*. [doi:10.1038/s41562-019-0566-x](https://doi.org/10.1038/s41562-019-0566-x)
  Genomic SEM is family (a) with an explicit factor model imposed on the genetic covariance matrix; its common-factor GWAS yields a single score, and QSNP is its homogeneity diagnostic — the genomic-SEM analogue of MTAG's maxFDR.
- **Ruan Y et al.** (2022). Improving polygenic prediction in ancestrally diverse populations. *Nature Genetics 54, 573-580*. [doi:10.1038/s41588-022-01054-7](https://doi.org/10.1038/s41588-022-01054-7)
  PRS-CSx is the clearest published hybrid: a shared prior couples effects across GWAS (family a), then the resulting population-specific scores are combined with weights learned in a validation set (family b), with an inverse-variance option (family c).
- **Weissbrod O et al.** (2022). Leveraging fine-mapping and multipopulation training data to improve cross-population polygenic risk scores. *Nature Genetics 54(4), 450-458*. [doi:10.1038/s41588-022-01036-9](https://doi.org/10.1038/s41588-022-01036-9)
  PolyPred is family (a) followed by family (b): distinct per-variant predictors are built, then linearly combined with learned weights — the same two-stage architecture as PRS-CSx and CT-SLEB.
- **Albiñana C et al.** (2023). Multi-PGS enhances polygenic prediction by combining 937 polygenic scores. *Nature Communications 14, 4702*. [doi:10.1038/s41467-023-40330-w](https://doi.org/10.1038/s41467-023-40330-w)
  The estimator multipgs.multi_pgs_fit implements; the canonical multi-PGS / PGS-Catalog-stacking paper.
- **Momin MM, Lee S, Wray NR, Lee SH** (2023). Significance tests for R2 of out-of-sample prediction using polygenic scores. *American Journal of Human Genetics 110(2):349-358*. [doi:10.1016/j.ajhg.2023.01.004](https://doi.org/10.1016/j.ajhg.2023.01.004)
  Justifies multipgs' insistence on reporting an interval, and is the analytic alternative to `evaluate`'s bootstrap -- including the covariance needed to compare two scores in the same cohort.
- **Truong B et al.** (2024). Integrative polygenic risk score improves the prediction accuracy of complex traits and diseases. *Cell Genomics 4(4), 100523*. [doi:10.1016/j.xgen.2024.100523](https://doi.org/10.1016/j.xgen.2024.100523)
  PRSmix/PRSmix+ is the closest published sibling of multipgs: PRSmix combines scores of the target trait, PRSmix+ adds genetically correlated traits — exactly the meta_pgs / multi_pgs_fit split, but with learned weights on both sides.
- **Mbebi AJ, Mercado F, Hobby D, Tong H, Nikoloski Z** (2025). Advances in multi-trait genomic prediction approaches: classification, comparative analysis, and perspectives. *Briefings in Bioinformatics 26(3), bbaf211*. [doi:10.1093/bib/bbaf211](https://doi.org/10.1093/bib/bbaf211)
  A recent taxonomy of multi-trait genomic prediction models; useful as a pointer for readers, but note it is written from the crop-breeding side, where phenotypes on all traits are measured in the same individuals — the assumption human multi-PGS specifically drops.

## Penalized regression and model selection

The estimator behind `multi_pgs_fit`, and the model-selection theory it depends on.

- **Ragnar Frisch, Frederick V. Waugh** (1933). Partial Time Regressions as Compared with Individual Trends. *Econometrica 1(4):387–401*. [doi:10.2307/1907330](https://doi.org/10.2307/1907330)
  The first half of the Frisch–Waugh–Lovell result that makes multipgs's unpenalized covariate columns equivalent to residualising in the Gaussian case.
- **Michael C. Lovell** (1963). Seasonal Adjustment of Economic Time Series and Multiple Regression Analysis. *Journal of the American Statistical Association 58(304):993–1010*. [doi:10.1080/01621459.1963.10480682](https://doi.org/10.1080/01621459.1963.10480682)
  The general statement of the partitioned-regression theorem invoked by docs/algorithm.md and by tests/test_coord.py::test_unpenalized_columns_equal_frisch_waugh_lovell.
- **Arthur E. Hoerl, Robert W. Kennard** (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. *Technometrics 12(1):55–67*. [doi:10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634)
  The alpha=0 end of multipgs's elastic-net dial, and the reason lambda_max is infinite there (no finite penalty zeroes a ridge solution).
- **M. Stone** (1974). Cross-Validatory Choice and Assessment of Statistical Predictions. *Journal of the Royal Statistical Society: Series B (Methodological) 36(2):111–133 (read with discussion; the discussion runs to p. 147)*. [doi:10.1111/j.2517-6161.1974.tb00994.x](https://doi.org/10.1111/j.2517-6161.1974.tb00994.x)
  The origin of the choice-versus-assessment distinction that the whole CMSA honesty discussion rests on.
- **Bradley Efron** (1986). How Biased is the Apparent Error Rate of a Prediction Rule?. *Journal of the American Statistical Association 81(394):461–470*. [doi:10.1080/01621459.1986.10478291](https://doi.org/10.1080/01621459.1986.10478291)
  The covariance-penalty formulation of optimism: the quantitative statement of why apparent error understates true error by an amount proportional to how much the fit chases its own response.
- **David L. Donoho, Iain M. Johnstone** (1994). Ideal spatial adaptation by wavelet shrinkage. *Biometrika 81(3):425–455*. [doi:10.1093/biomet/81.3.425](https://doi.org/10.1093/biomet/81.3.425)
  The soft-thresholding operator S(z,gamma)=sign(z)(|z|-gamma)_+ that the coordinate update reduces to, and its risk properties in the orthonormal case.
- **Leo Breiman** (1996). Bagging Predictors. *Machine Learning 24(2):123–140*. [doi:10.1023/A:1018054314350](https://doi.org/10.1023/A:1018054314350)
  The variance-reduction argument for averaging models fitted on resamples — the family of reasoning CMSA's coefficient averaging belongs to.
- **Robert Tibshirani** (1996). Regression Shrinkage and Selection Via the Lasso. *Journal of the Royal Statistical Society: Series B (Methodological) 58(1):267–288*. [doi:10.1111/j.2517-6161.1996.tb02080.x](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x)
  The L1 penalty that multipgs uses by default (alpha=1.0) and the source of exact-zero coefficients in `MultiPGSFit.beta`.
- **Bradley Efron, Robert Tibshirani** (1997). Improvements on Cross-Validation: The .632+ Bootstrap Method. *Journal of the American Statistical Association 92(438):548–560*. [doi:10.1080/01621459.1997.10474007](https://doi.org/10.1080/01621459.1997.10474007)
  One of the standard remedies for optimistic error estimates, and a useful contrast: it corrects fitting optimism, not selection optimism.
- **Christophe Ambroise, Geoffrey J. McLachlan** (2002). Selection bias in gene extraction on the basis of microarray gene-expression data. *Proceedings of the National Academy of Sciences 99(10):6562–6566*. [doi:10.1073/pnas.102102699](https://doi.org/10.1073/pnas.102102699)
  The earlier and equally direct statement of the same problem: cross-validation that is internal to a selection step does not measure generalisation.
- **Peter Bühlmann, Bin Yu** (2002). Analyzing bagging. *The Annals of Statistics 30(4)*. [doi:10.1214/aos/1031689014](https://doi.org/10.1214/aos/1031689014)
  Makes the CMSA averaging argument precise for exactly the estimator at issue: bagging smooths a hard-threshold (selection) rule and cuts variance where the rule is unstable.
- **Zou H, Hastie T** (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society Series B 67(2), 301-320*. [doi:10.1111/j.1467-9868.2005.00503.x](https://doi.org/10.1111/j.1467-9868.2005.00503.x)
  The penalty in multipgs._coord; its grouping property is the specific reason an elastic net rather than a lasso is right for a panel containing many near-duplicate scores of the same trait.
- **Hui Zou** (2006). The Adaptive Lasso and Its Oracle Properties. *Journal of the American Statistical Association 101(476):1418–1429*. [doi:10.1198/016214506000000735](https://doi.org/10.1198/016214506000000735)
  The theoretical justification multipgs invokes for per-column penalty factors (`penalty_factor`, `penalty_from_accuracy`), and the reason those factors must be interpreted as a prior rather than as an oracle guarantee here.
- **Sudhir Varma, Richard Simon** (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics 7:91*. [doi:10.1186/1471-2105-7-91](https://doi.org/10.1186/1471-2105-7-91)
  The canonical demonstration that tuning and assessing on the same cross-validation is optimistic, and that nested CV fixes it — the exact failure `_honest_cv_loss` is built to avoid.
- **Hui Zou, Trevor Hastie, Robert Tibshirani** (2007). On the “degrees of freedom” of the lasso. *The Annals of Statistics 35(5)*. [doi:10.1214/009053607000000127](https://doi.org/10.1214/009053607000000127)
  Supplies the exact degrees-of-freedom count that makes the optimism argument quantitative — and makes clear it holds at fixed lambda, not at a lambda chosen from the data.
- **Jerome Friedman, Trevor Hastie, Holger Höfling, Robert Tibshirani** (2007). Pathwise coordinate optimization. *The Annals of Applied Statistics 1(2):302–332*. [doi:10.1214/07-AOAS131](https://doi.org/10.1214/07-AOAS131)
  Establishes that one-at-a-time coordinate descent solves lasso/elastic-net problems and is competitive with LARS — the justification for the sweep-based solver in _coord.py.
- **Ryan J. Tibshirani, Robert Tibshirani** (2009). A bias correction for the minimum error rate in cross-validation. *The Annals of Applied Statistics 3(2)*. [doi:10.1214/08-AOAS224](https://doi.org/10.1214/08-AOAS224)
  A direct estimator of the winner's-curse component in min-over-lambda CV error, computable from the per-fold CV curves multipgs already stores in `loss_table`.
- **Trevor Hastie, Robert Tibshirani, Jerome Friedman** (2009). The Elements of Statistical Learning: Data Mining, Inference, and Prediction (2nd edition). *Springer, New York*. [doi:10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  Textbook home of the optimism decomposition, the K-fold training-size bias, the one-standard-error rule, and the 'wrong way / right way' cross-validation example.
- **Friedman J, Hastie T, Tibshirani R** (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software 33(1)*. [doi:10.18637/jss.v033.i01](https://doi.org/10.18637/jss.v033.i01)
  The coordinate-descent path algorithm that multipgs._coord reimplements (covariance updates for Gaussian, IRLS for binomial), and the glmnet package Albiñana et al. actually used.
- **Gavin C. Cawley, Nicola L. C. Talbot** (2010). On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. *Journal of Machine Learning Research 11(70):2079–2107*.
  Explains why the variance of the selection criterion — not just its bias — is what drives optimism, which is exactly why per-fold selection on a few dozen individuals is so badly behaved in multi_pgs_fit.
- **Nicolai Meinshausen, Peter Bühlmann** (2010). Stability Selection. *Journal of the Royal Statistical Society Series B: Statistical Methodology 72(4):417–473*. [doi:10.1111/j.1467-9868.2010.00740.x](https://doi.org/10.1111/j.1467-9868.2010.00740.x)
  The alternative use of the same resampling information: aggregate selection frequencies rather than coefficients. Useful contrast for interpreting `MultiPGSFit.selected()`.
- **Robert Tibshirani, Jacob Bien, Jerome Friedman, Trevor Hastie, Noah Simon, Jonathan Taylor, Ryan J. Tibshirani** (2012). Strong Rules for Discarding Predictors in Lasso-Type Problems. *Journal of the Royal Statistical Society Series B: Statistical Methodology 74(2):245–266 (published online 2011)*. [doi:10.1111/j.1467-9868.2011.01004.x](https://doi.org/10.1111/j.1467-9868.2011.01004.x)
  The standard screening rule that _coord.py deliberately does not use — its active-set cycling is the older, always-safe Friedman-2010 strategy; worth naming the difference in the algorithm docs.
- **Bradley Efron** (2014). Estimation and Accuracy After Model Selection. *Journal of the American Statistical Association 109(507):991–1007*. [doi:10.1080/01621459.2013.823775](https://doi.org/10.1080/01621459.2013.823775)
  States the problem CMSA's averaging quietly solves: after selection the estimator is a discontinuous function of the data, and bagging restores smoothness and makes accuracy assessment tractable.
- **Trevor Hastie, Robert Tibshirani, Martin Wainwright** (2015). Statistical Learning with Sparsity: The Lasso and Generalizations. *Chapman and Hall/CRC*. [doi:10.1201/b18401](https://doi.org/10.1201/b18401)
  Reference for the KKT/subgradient conditions, uniqueness of the lasso solution, and coordinate-descent convergence for separable non-smooth penalties.
- **Timothy Shin Heng Mak, Robert Milan Porsch, Shing Wan Choi, Xueya Zhou, Pak Chung Sham** (2017). Polygenic scores via penalized regression on summary statistics. *Genetic Epidemiology 41(6):469–480*. [doi:10.1002/gepi.22050](https://doi.org/10.1002/gepi.22050)
  The summary-statistic form of the same penalized-regression objective, and the canonical answer to 'how do you tune lambda with no validation phenotype' — the question CMSA answers differently.
- **Florian Privé, Hugues Aschard, Andrey Ziyatdinov, Michael G. B. Blum** (2018). Efficient analysis of large-scale genome-wide data with two R packages: bigstatsr and bigsnpr. *Bioinformatics 34(16):2781–2787*. [doi:10.1093/bioinformatics/bty185](https://doi.org/10.1093/bioinformatics/bty185)
  The software home of big_spLinReg/big_spLogReg, whose CMSA loop, K=10, n.abort=10, alphas=1, covar.train and pf.X defaults multipgs mirrors one-for-one.
- **Florian Privé, Hugues Aschard, Michael G. B. Blum** (2019). Efficient Implementation of Penalized Regression for Genetic Risk Prediction. *Genetics 212(1):65–74*. [doi:10.1534/genetics.119.302019](https://doi.org/10.1534/genetics.119.302019)
  The source of Cross-Model Selection and Averaging (CMSA), which multi_pgs_fit reimplements, and the reference for penalized regression beating C+T for genetic risk prediction.
- **Privé F, Vilhjálmsson BJ, Aschard H, Blum MGB** (2019). Making the most of clumping and thresholding for polygenic scores. *American Journal of Human Genetics 105(6), 1213-1221*. [doi:10.1016/j.ajhg.2019.11.001](https://doi.org/10.1016/j.ajhg.2019.11.001)
  SCT is family (b) applied to thousands of scores of a SINGLE trait — the score-stacking idea one step before Albiñana et al. generalised it across traits.
- **Junyang Qian et al.** (2020). A fast and scalable framework for large-scale and ultrahigh-dimensional sparse regression with application to the UK Biobank. *PLOS Genetics 16(10):e1009141*. [doi:10.1371/journal.pgen.1009141](https://doi.org/10.1371/journal.pgen.1009141)
  The other production-scale lasso for genetic prediction (snpnet), and the contrasting engineering choice: batch screening over millions of variants rather than a dense Gram over a few thousand scores.

## Accuracy, power and scales

What an accuracy number means, and the scale it lives on.

- **Dempster ER, Lerner IM** (1950). Heritability of threshold characters. *Genetics 35(2):212-236*. [doi:10.1093/genetics/35.2.212](https://doi.org/10.1093/genetics/35.2.212)
  The original observed-to-liability scaling h2_obs = h2_liab * z^2/[K(1-K)]; the leading factor of the Lee et al. transformation is this result, with the ascertainment factor bolted on.
- **Falconer DS** (1965). The inheritance of liability to certain diseases, estimated from the incidence among relatives. *Annals of Human Genetics 29(1):51-76*. [doi:10.1111/j.1469-1809.1965.tb00500.x](https://doi.org/10.1111/j.1469-1809.1965.tb00500.x)
  The liability threshold model underlying every liability-scale quantity in `multipgs.metrics`, including the definitions of t, z and the mean liability of cases.
- **Cox DR, Snell EJ** (1989). Analysis of Binary Data, 2nd edition. *Chapman and Hall, London (ISBN 978-0-412-30620-4)*.
  Source of the Cox-Snell R2 = 1 - (L_null/L_full)^(2/n) that Nagelkerke rescales and that multipgs computes internally.
- **Nagelkerke NJD** (1991). A note on a general definition of the coefficient of determination. *Biometrika 78(3):691-692*. [doi:10.1093/biomet/78.3.691](https://doi.org/10.1093/biomet/78.3.691)
  `multipgs.metrics.nagelkerke_r2` implements exactly this rescaling of the Cox-Snell R2.
- **Daetwyler HD, Villanueva B, Woolliams JA** (2008). Accuracy of predicting the genetic risk of disease using a genome-wide approach. *PLoS ONE 3(10), e3395*. [doi:10.1371/journal.pone.0003395](https://doi.org/10.1371/journal.pone.0003395)
  The expected-accuracy bound behind multipgs.daetwyler_r2, and hence behind every derived weight meta_pgs produces.
- **Goddard M** (2009). Genomic selection: prediction of accuracy and maximisation of long term response. *Genetica 136(2), 245-257*. [doi:10.1007/s10709-008-9308-0](https://doi.org/10.1007/s10709-008-9308-0)
  Origin of the effective number of independent chromosome segments M_e, the quantity that Daetwyler's M and Maier et al.'s ~60,000 both stand for and that multipgs replaces with n_variants * p from the fitted polygenicity.
- **Daetwyler HD, Pong-Wong R, Villanueva B, Woolliams JA** (2010). The impact of genetic architecture on genome-wide evaluation methods. *Genetics 185(3):1021-1031*. [doi:10.1534/genetics.110.116855](https://doi.org/10.1534/genetics.110.116855)
  Shows the deterministic accuracy equation behaves differently for a linear (GBLUP-like) versus a variable-selection (BayesB-like) estimator, which is exactly the difference between M = all variants and M = n_variants * p.
- **Wray NR, Yang J, Goddard ME, Visscher PM** (2010). The genetic interpretation of area under the ROC curve in genomic profiling. *PLoS Genetics 6(2):e1000864*. [doi:10.1371/journal.pgen.1000864](https://doi.org/10.1371/journal.pgen.1000864)
  Explains why `multipgs.metrics.auc` needs no ascertainment correction while `r2` does, and gives the AUC ceiling that bounds any binary PGS.
- **Lee SH, Wray NR, Goddard ME, Visscher PM** (2011). Estimating missing heritability for disease from genome-wide association studies. *American Journal of Human Genetics 88(3):294-305*. [doi:10.1016/j.ajhg.2011.02.002](https://doi.org/10.1016/j.ajhg.2011.02.002)
  Supplies the C factor -- the leading term of `multipgs.metrics.liability_r2` -- and is the transformation ldpred3.h2_liability applies, as multipgs' docstring states.
- **Lee SH, Goddard ME, Wray NR, Visscher PM** (2012). A better coefficient of determination for genetic profile analysis. *Genetic Epidemiology 36(3), 214-224*. [doi:10.1002/gepi.21614](https://doi.org/10.1002/gepi.21614)
  The liability-scale R2 in multipgs.metrics.liability_r2, including the ascertainment (theta) correction — without which case/control multi-PGS accuracies are not comparable across studies.
- **Chatterjee N, Wheeler B, Sampson J, Hartge P, Chanock SJ, Park JH** (2013). Projecting the performance of risk prediction based on polygenic analyses of genome-wide association studies. *Nature Genetics 45(4), 400-405*. [doi:10.1038/ng.2579](https://doi.org/10.1038/ng.2579)
  The complementary projection framework to Daetwyler: it makes the effect-size distribution explicit, which is where the Daetwyler bound's optimism comes from.
- **Dudbridge F** (2013). Power and predictive accuracy of polygenic risk scores. *PLoS Genetics 9(3), e1003348*. [doi:10.1371/journal.pgen.1003348](https://doi.org/10.1371/journal.pgen.1003348)
  The reference derivation of PGS accuracy as a function of discovery and target sample sizes; underwrites the claim that single-trait scores are weak in exactly the regime where multi-PGS pays.
- **Wray NR, Yang J, Hayes BJ, Price AL, Goddard ME, Visscher PM** (2013). Pitfalls of predicting complex traits from SNPs. *Nature Reviews Genetics 14(7):507-515*. [doi:10.1038/nrg3457](https://doi.org/10.1038/nrg3457)
  Supplies the failure modes multipgs' README and metrics module are organised around -- in particular that no internal cross-validation can detect discovery/target sample overlap.
- **Wray NR, Kemper KE, Hayes BJ, Goddard ME, Visscher PM** (2019). Complex Trait Prediction from Genome Data: Contrasting EBV in Livestock to PRS in Humans: Genomic Prediction. *Genetics 211(4):1131-1141*. [doi:10.1534/genetics.119.301859](https://doi.org/10.1534/genetics.119.301859)
  The clearest modern statement of the Daetwyler-style expected-R2 equation in human PRS terms, including why M_e (not the causal count) governs, and why GWAS sample size is the binding constraint.
- **Choi SW, Mak TS-H, O'Reilly PF** (2020). Tutorial: a guide to performing polygenic risk score analyses. *Nature Protocols 15(9):2759-2772*. [doi:10.1038/s41596-020-0353-1](https://doi.org/10.1038/s41596-020-0353-1)
  The citable source for the two reporting rules multipgs' metrics module enforces: report incremental R2 when covariates are present, and use Lee et al.'s liability-scale R2 rather than Nagelkerke for case/control.
- **David Keetae Park et al.** (2023). Overestimated prediction using polygenic prediction derived from summary statistics. *BMC Genomic Data 24:52*. [doi:10.1186/s12863-023-01151-4](https://doi.org/10.1186/s12863-023-01151-4)
  A concrete, quantified demonstration of overlap inflation with the same overlap removed and retained — the number to cite when the docs claim overlap is the failure that matters most.

## Summary-statistic methods, LD and architecture

How the input scores are built, and the architecture parameters `multipgs.architecture` screens on.

- **Doug Speed, Gibran Hemani, Michael R. Johnson, David J. Balding** (2012). Improved heritability estimation from genome-wide SNPs. *American Journal of Human Genetics 91(6):1011-1021*. [doi:10.1016/j.ajhg.2012.10.010](https://doi.org/10.1016/j.ajhg.2012.10.010)
  The LDAK model, where the alpha exponent relating allele frequency to expected per-SNP heritability first appears; it is the same alpha LDpred2-auto now infers.
- **Brendan K. Bulik-Sullivan et al.** (2015). LD Score regression distinguishes confounding from polygenicity in genome-wide association studies. *Nature Genetics 47(3):291-295*. [doi:10.1038/ng.3211](https://doi.org/10.1038/ng.3211)
  Univariate LDSC gives the h2 that feeds the Daetwyler bound when an LDpred2-auto fit is unavailable, and its intercept is the standard diagnostic for stratification and cryptic relatedness in a discovery GWAS.
- **Bulik-Sullivan B, et al. (ReproGen Consortium, Psychiatric Genomics Consortium, and Genetic Consortium for Anorexia Nervosa of the Wellcome Trust Case Control Consortium 3)** (2015). An atlas of genetic correlations across human diseases and traits. *Nature Genetics 47(11), 1236-1241*. [doi:10.1038/ng.3406](https://doi.org/10.1038/ng.3406)
  The source of the genetic correlation estimates that every derived-weight (family c) method needs — and, by its coverage limits, the reason a 937-score PGS Catalog panel has to use learned weights instead.
- **Vilhjálmsson BJ, Yang J, Finucane HK, Gusev A, Lindström S, Ripke S, et al.** (2015). Modeling linkage disequilibrium increases accuracy of polygenic risk scores. *American Journal of Human Genetics 97(4), 576-592*. [doi:10.1016/j.ajhg.2015.09.001](https://doi.org/10.1016/j.ajhg.2015.09.001)
  The original LDpred; the single-trait construction step that every score entering a multi-PGS panel is the output of.
- **Doug Speed, Na Cai, UCLEB Consortium, Michael R. Johnson, Sergey Nejentsev, David J. Balding** (2017). Reevaluation of SNP heritability in complex human traits. *Nature Genetics 49(7):986-992*. [doi:10.1038/ng.3865](https://doi.org/10.1038/ng.3865)
  Derives, from 42 traits, how heritability varies with minor allele frequency, LD and genotype certainty — the empirical case that alpha = -1 (the implicit GCTA/LDpred assumption) is wrong for most traits.
- **Jian Zeng et al.** (2018). Signatures of negative selection in the genetic architecture of human complex traits. *Nature Genetics 50(5):746-753*. [doi:10.1038/s41588-018-0101-4](https://doi.org/10.1038/s41588-018-0101-4)
  The BayesS model that jointly estimates h2, polygenicity and the effect-size/MAF relationship — the direct antecedent of what LDpred2-auto now does from summary statistics alone.
- **Zhang Y, Qi G, Park JH, Chatterjee N** (2018). Estimation of complex effect-size distributions using summary-level statistics from genome-wide association studies across 32 complex traits. *Nature Genetics 50(9):1318-1326*. [doi:10.1038/s41588-018-0193-x](https://doi.org/10.1038/s41588-018-0193-x)
  The successor to Chatterjee 2013 and the closest published analogue to what multipgs consumes: per-trait polygenicity and effect-size distribution inferred from summary statistics, then used to project PGS accuracy.
- **Armin P. Schoech et al.** (2019). Quantification of frequency-dependent genetic architectures in 25 UK Biobank traits reveals action of negative selection. *Nature Communications 10:790*. [doi:10.1038/s41467-019-08424-6](https://doi.org/10.1038/s41467-019-08424-6)
  Fixes the sign convention and the empirical magnitude of alpha, so a reader can tell whether an LDpred2-auto alpha estimate is plausible.
- **Luke R. Lloyd-Jones et al.** (2019). Improved polygenic prediction by Bayesian multiple regression on summary statistics. *Nature Communications 10:5086*. [doi:10.1038/s41467-019-12653-0](https://doi.org/10.1038/s41467-019-12653-0)
  SBayesR, the mixture-of-normals member of the summary-statistic Bayesian family; useful for naming the alternatives to LDpred in the docs without overclaiming a ranking.
- **Tian Ge, Chia-Yen Chen, Yang Ni, Yen-Chen Anne Feng, Jordan W. Smoller** (2019). Polygenic prediction via Bayesian regression and continuous shrinkage priors. *Nature Communications 10:1776*. [doi:10.1038/s41467-019-09718-5](https://doi.org/10.1038/s41467-019-09718-5)
  PRS-CS, the continuous-shrinkage alternative to the point-normal prior; names the third branch of the summary-statistic Bayesian family alongside LDpred and SBayes*.
- **Doug Speed, John Holmes, David J. Balding** (2020). Evaluating and improving heritability models using summary statistics. *Nature Genetics 52(4):458-462*. [doi:10.1038/s41588-020-0600-y](https://doi.org/10.1038/s41588-020-0600-y)
  The paper behind LDAK's 'human default' heritability model; the LDAK documentation states that model as E[h2_j] = tau1 * w_j * [f_j(1-f_j)]^(1+alpha) with alpha = -0.25, which is the reference point for interpreting an LDpred2-auto alpha estimate.
- **Florian Privé, Julyan Arbel, Bjarni J. Vilhjálmsson** (2020). LDpred2: better, faster, stronger. *Bioinformatics 36(22-23):5424–5431*. [doi:10.1093/bioinformatics/btaa1029](https://doi.org/10.1093/bioinformatics/btaa1029)
  The method behind the input scores in the multi-PGS setting, and the third answer to hyper-parameter selection: estimate the parameters inside the model instead of tuning them.
- **Jian Zeng et al.** (2021). Widespread signatures of natural selection across human complex traits and functional genomic categories. *Nature Communications 12:1164*. [doi:10.1038/s41467-021-21446-3](https://doi.org/10.1038/s41467-021-21446-3)
  SBayesS, the summary-statistic method whose S parameter is the same quantity LDpred2-auto calls alpha — the LDpred2-auto paper cites it as the model its Equation 2 is similar to.
- **Zijie Zhao et al.** (2021). PUMAS: fine-tuning polygenic risk scores with GWAS summary statistics. *Genome Biology 22:257*. [doi:10.1186/s13059-021-02479-9](https://doi.org/10.1186/s13059-021-02479-9)
  The principled alternative to pseudovalidation: manufacture genuinely independent pseudo-training/pseudo-validation summary statistics so that tuning and assessment are separated even without individual-level data.
- **Florian Privé, Julyan Arbel, Hugues Aschard, Bjarni J. Vilhjálmsson** (2022). Identifying and correcting for misspecifications in GWAS summary statistics and polygenic scores. *Human Genetics and Genomics Advances 3(4):100136*. [doi:10.1016/j.xhgg.2022.100136](https://doi.org/10.1016/j.xhgg.2022.100136)
  The QC theory behind why a summary-statistic Bayesian fit fails to converge: INFO scores, allele frequencies and per-variant N disagreeing with the LD reference is the usual cause, which is exactly what multipgs' chain-convergence gate detects.
- **Florian Privé, Clara Albiñana, Julyan Arbel, Bogdan Pasaniuc, Bjarni J. Vilhjálmsson** (2023). Inferring disease architecture and predictive ability with LDpred2-auto. *American Journal of Human Genetics 110(12):2042-2055*. [doi:10.1016/j.ajhg.2023.10.010](https://doi.org/10.1016/j.ajhg.2023.10.010)
  Supplies every quantity multipgs.architecture consumes: h2, polygenicity p, the alpha parameter, the inferred out-of-sample r2, and the multi-chain protocol whose convergence count Hansen et al.'s screen thresholds.
- **Xu C, Ganesh SK, Zhou X** (2023). mtPGS: Leverage multiple correlated traits for accurate polygenic score construction. *American Journal of Human Genetics 110(10), 1673-1689*. [doi:10.1016/j.ajhg.2023.08.016](https://doi.org/10.1016/j.ajhg.2023.08.016)
  A recent family-(a) method: borrows SNP effect-size similarity between a target trait and relevant traits from summary statistics alone, so it is the natural 'combine before' comparator for multi_pgs_fit.
- **Seokho Jeong et al.** (2025). Addressing overfitting bias due to sample overlap in polygenic risk scoring. *Alzheimer's & Dementia 21(4):e70109*. [doi:10.1002/alz.70109](https://doi.org/10.1002/alz.70109)
  Shows the magnitude of overlap inflation in a real IGAP/ADNI pair and that an explicit correction, not resampling within the target, is what removes it.

## PGS Catalog, practice and ancestry

Operational practice, and the two things that most often invalidate a reported number.

- **Alicia R. Martin et al.** (2017). Human Demographic History Impacts Genetic Risk Prediction across Diverse Populations. *American Journal of Human Genetics 100(4):635-649*. [doi:10.1016/j.ajhg.2017.03.004](https://doi.org/10.1016/j.ajhg.2017.03.004)
  Shows that portability failure is not only a loss of discrimination but a directional bias in the mean score — the reason a raw multi-PGS value must not be read as a calibrated risk across ancestry groups.
- **Inouye M, Abraham G, Nelson CP, Wood AM, Sweeting MJ, Dudbridge F, et al. (UK Biobank CardioMetabolic Consortium CHD Working Group)** (2018). Genomic risk prediction of coronary artery disease in 480,000 adults: implications for primary prevention. *Journal of the American College of Cardiology 72(16), 1883-1893*. [doi:10.1016/j.jacc.2018.07.079](https://doi.org/10.1016/j.jacc.2018.07.079)
  metaGRS: an influential clinical instance of family (b) — several same-trait genomic risk scores combined into one deployed 1.7M-variant weight file, which is precisely what combine_weights produces.
- **Alicia R. Martin, Masahiro Kanai, Yoichiro Kamatani, Yukinori Okada, Benjamin M. Neale, Mark J. Daly** (2019). Clinical use of current polygenic risk scores may exacerbate health disparities. *Nature Genetics 51(4):584-591*. [doi:10.1038/s41588-019-0379-x](https://doi.org/10.1038/s41588-019-0379-x)
  The canonical statement of the portability limit: a panel of PGS Catalog scores is overwhelmingly European-derived, so a multi-PGS stack inherits that limit no matter how well the combination is fitted.
- **Cai M, Xiao J, Zhang S, Wan X, Zhao H, Chen G, Yang C** (2021). A unified framework for cross-population trait prediction by leveraging the genetic correlation of polygenic traits. *American Journal of Human Genetics 108(4), 632-655*. [doi:10.1016/j.ajhg.2021.03.002](https://doi.org/10.1016/j.ajhg.2021.03.002)
  XPA/XPASS: the cross-population instance of the same problem, where the borrowed 'trait' is the same trait in another ancestry and rG is the trans-ancestry genetic correlation.
- **Lambert SA et al.** (2021). The Polygenic Score Catalog as an open database for reproducibility and systematic evaluation. *Nature Genetics 53(4), 420-425*. [doi:10.1038/s41588-021-00783-5](https://doi.org/10.1038/s41588-021-00783-5)
  The resource that makes score-level stacking possible at all: multipgs.catalog reads its scoring-file format, and panel_from_catalog is built around it.
- **Florian Privé, Hugues Aschard, Shai Carmi, Lasse Folkersen, Clive Hoggart, Paul F. O'Reilly, Bjarni J. Vilhjálmsson** (2022). Portability of 245 polygenic scores when derived from the UK Biobank and applied to 9 ancestry groups from the same cohort. *American Journal of Human Genetics 109(1):12-23*. [doi:10.1016/j.ajhg.2021.11.008](https://doi.org/10.1016/j.ajhg.2021.11.008)
  Isolates portability decay from cohort and genotyping confounding by training and testing inside one cohort, and shows the decay is continuous in genetic distance — including within Europe, which matters for a panel assembled from many European sub-cohorts.
- **Kullo IJ, Lewis CM, Inouye M, Martin AR, Ripatti S, Chatterjee N** (2022). Polygenic scores in biomedical research. *Nature Reviews Genetics 23(9), 524-532 — Viewpoint (multi-author expert-opinion piece), not a Review*. [doi:10.1038/s41576-022-00470-z](https://doi.org/10.1038/s41576-022-00470-z)
  A short, citable review for the docs' 'before trusting a number' section: reporting standards, portability, and the interpretation limits that apply to any combined score.
- **Zhang H et al.** (2023). A new method for multiancestry polygenic prediction improves performance across diverse populations. *Nature Genetics 55(10), 1757-1768*. [doi:10.1038/s41588-023-01501-z](https://doi.org/10.1038/s41588-023-01501-z)
  CT-SLEB: clumping+thresholding, empirical Bayes and superlearning stacked together — the largest-scale demonstration that a learned combination stage on top of per-variant methods is what actually delivers the accuracy.
- **Samuel A. Lambert et al.** (2024). Enhancing the Polygenic Score Catalog with tools for score calculation and ancestry normalization. *Nature Genetics 56(10):1989-1994*. [doi:10.1038/s41588-024-01937-x](https://doi.org/10.1038/s41588-024-01937-x)
  Defines the reference implementation (pgsc_calc) of the scoring step multipgs performs itself, including the ancestry-normalisation step multipgs does not do — worth naming explicitly in the docs so users know what is out of scope.
- **Hansen O et al.** (2026). Mapping genetic architecture of thousands of complex traits using GWAS summary statistics. *Research Square (preprint, not peer reviewed)*. [doi:10.21203/rs.3.rs-9415305/v1](https://doi.org/10.21203/rs.3.rs-9415305/v1)
  The source of multipgs.architecture.screen's inclusion gates and of the sqrt(n_eff) meta-PGS rule; also the source of taking M = n_variants * p from the fitted polygenicity rather than a fixed M_e.
