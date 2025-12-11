# Flow-induced reprogramming of endothelial cells in atherosclerosis 

Ian A. Tamargo ${ }^{1,2,4}$, Kyung In Baek ${ }^{1,4}$, Yerin Kim ${ }^{1}$, Christian Park ${ }^{1}$ \& Hanjoong Jo ${ }^{1,2,3}$


#### Abstract

Atherosclerotic diseases such as myocardial infarction, ischaemic stroke and peripheral artery disease continue to be leading causes of death worldwide despite the success of treatments with cholesterol-lowering drugs and drug-eluting stents, raising the need to identify additional therapeutic targets. Interestingly, atherosclerosis preferentially develops in curved and branching arterial regions, where endothelial cells are exposed to disturbed blood flow with characteristic low-magnitude oscillatory shear stress. By contrast, straight arterial regions exposed to stable flow, which is associated with high-magnitude, unidirectional shear stress, are relatively well protected from the disease through sheardependent, atheroprotective endothelial cell responses. Flow potently regulates structural, functional, transcriptomic, epigenomic and metabolic changes in endothelial cells through mechanosensors and mechanosignal transduction pathways. A study using single-cell RNA sequencing and chromatin accessibility analysis in a mouse model of flow-induced atherosclerosis demonstrated that disturbed flow reprogrammes arterial endothelial cells in situ from healthy phenotypes to diseased ones characterized by endothelial inflammation, endothelial-to-mesenchymal transition, endothelial-to-immune cell-like transition and metabolic changes. In this Review, we discuss this emerging concept of disturbed-flow-induced reprogramming of endothelial cells (FIRE) as a potential pro-atherogenic mechanism. Defining the flow-induced mechanisms through which endothelial cells are reprogrammed to promote atherosclerosis is a crucial area of research that could lead to the identification of novel therapeutic targets to combat the high prevalence of atherosclerotic disease.


[^0]
## Sections

Introduction
Atherosclerosis preferentially develops at sites of disturbed flow

Blood flow regulates endothelial function

Mechanosensors and mechanotransduction

Omics approaches to study endothelial cells

Disturbed-flow-induced reprogramming of endothelial cells

Therapeutic implications in atherosclerosis

Conclusions


[^0]:    ${ }^{1}$ Wallace H. Coulter Department of Biomedical Engineering, Emory University and Georgia Institute of Technology, Atlanta, GA, USA. ${ }^{2}$ Molecular and Systems Pharmacology Program, Emory University, Atlanta, GA, USA. ${ }^{3}$ Department of Medicine, Emory University School, Atlanta, GA, USA. ${ }^{4}$ These authors contributed equally: Ian A. Tamargo; Kyung In Baek. $\square$ e-mail: hjo@emory.edu

# Review article 

## Key points

- Atherosclerosis preferentially develops in curved and branching regions of arteries, which are sites of disturbed blood flow and low-magnitude oscillatory shear stress.
- Disturbed flow delivers low-magnitude oscillatory shear stress to endothelial cells, which causes endothelial cells to adopt pro-atherogenic functions and gene transcription programmes.
- Endothelial cells detect shear stress magnitudes and patterns through mechanosensory proteins and organelles and transmit these signals into intracellular changes via mechanotransduction pathways.
- Advances in omics approaches and experimental models have helped to identify numerous novel potential therapeutic targets for atherosclerosis.
- Disturbed-flow-induced reprogramming of endothelial cells (which we term FIRE) promotes endothelial inflammation, endothelial-tomesenchymal transition and endothelial-to-immune-cell-like transition during atherogenesis.
- Genes, proteins and pathways involved in FIRE are promising targets for anti-atherogenic therapies.


## Introduction

Atherosclerosis is a multifactorial and chronic inflammatory disease of the arteries, in which fibrofatty plaques develop in the arterial wall ${ }^{1}$. As advanced plaques develop, the arterial wall stiffens, the arterial lumen narrows and, occasionally, plaques rupture or are eroded, resulting in severe clinical consequences, including myocardial infarction, ischaemic stroke and peripheral artery disease, which are the leading causes of death worldwide ${ }^{2}$.

Dysfunction and inflammation of endothelial cells have a crucial role in the initiation and progression of atherosclerosis ${ }^{3}$. Endothelial cells lining the inner layer of the blood vessels are in direct contact with the blood and become dysfunctional and inflamed in response to various risk factors, such as hypercholesterolaemia, diabetes mellitus, hypertension, smoking and ageing, especially at specific atherosclerosisprone regions associated with disturbed blood flow. At these sites of endothelial inflammation, circulating monocytes bind to endothelial cells and transmigrate into the subendothelial space, differentiating into macrophages. These regions also show increased permeability to circulating LDL cholesterol, which becomes oxidized in the subendothelial space and is ingested by nearby macrophages, thereby promoting foam cell development and triggering a vicious cycle of inflammation and macrophage accumulation ${ }^{1}$. In addition, vascular smooth muscle cells (VSMCs) in these regions transdifferentiate into synthetic phenotypes and migrate to the subendothelial layer and proliferate, contributing to arterial wall thickening ${ }^{4}$. In addition, some VSMCs transdifferentiate into foam cells ${ }^{5}$, eventually leading to the formation of fatty streaks and fibrofatty plaques (Fig. 1a). As atherosclerotic plaques grow outwardly and inwardly, mature and progress, some plaques rupture, causing major cardiovascular events, such as myocardial infarction and stroke ${ }^{5,6}$.

Although atherosclerotic risk factors such as hypercholesterolaemia, hyperglycaemia and hypertension are systemic, plaques
preferentially develop in a focal manner in curved and branching regions of the arteries associated with disturbed blood flow ${ }^{7}$. Disturbed flow in these regions is characterized by the delivery of low-magnitude oscillatory shear stress (OSS) to the endothelial cell surface ${ }^{7-9}$. Endothelial cells detect various shear stress patterns and magnitudes through mechanosensing receptors (mechanosensors), which translate these mechanical cues into cell signalling (mechanosignal transduction) and subsequent structural and functional responses ${ }^{10}$. Advanced omics analyses have demonstrated that flow potently regulates nearly all facets of endothelial cell biology and pathobiology, from individual molecules and genes to structures and functions of the entire cell. Flow regulates endothelial cell transcriptomic and epigenomic landscapes at a genome-wide scale in vivo and in vitro, altering endothelial cell function, proliferation, survival and differentiation. Whereas stable blood flow, with the characteristic high-magnitude, unidirectional laminar shear stress (ULS) observed in straight, non-branching regions of the vasculature, promotes atheroprotective endothelial cell homeostasis, disturbed flow promotes pro-atherogenic endothelial cell responses, including endothelial dysfunction.

Although lipid-lowering drugs such as statins and PCSK9 inhibitors are highly efficient in reducing blood cholesterol levels and cardiovascular disease burden ${ }^{11}$, atherosclerotic diseases continue to be leading causes of death worldwide, highlighting the need for novel anti-atherogenic drugs targeting non-lipid, pro-atherogenic pathways. In this context, the CANTOS trial ${ }^{12}$ demonstrated that inhibition of vascular inflammation using the IL-1 $\beta$ inhibitor canakinumab significantly reduced atherothrombotic events in patients with previous myocardial infarction compared with placebo, in a cholesterolindependent manner. Although canakinumab was not approved by the FDA owing to an increase in fatal infections with canakinumab treatment in the trial, the findings demonstrated that targeting an inflammatory pathway could be a novel and effective anti-atherogenic therapy. Similarly, genes, proteins and pathways regulated by flow (flow-sensitive) that control pro-atherogenic endothelial cell dysfunction and inflammation could be promising novel therapeutic targets for atherosclerosis. To this end, in this Review, we discuss the current literature on flow-sensitive genes, proteins and pathways, including the emerging concept of disturbed-flow-induced reprogramming of endothelial cells (FIRE), involved in endothelial dysfunction and atherosclerosis.

## Atherosclerosis preferentially develops at sites of disturbed flow

## Vascular haemodynamics

The vascular endothelium is in direct contact with the blood in the arterial lumen and forms a protective barrier between the blood and the outer vascular wall ${ }^{13,14}$. The vascular endothelium is constantly exposed to haemodynamic forces: normal (transmural) stress and circumferential stress in the arterial wall resulting from blood pressure, and tangential shear stress on the endothelial surface due to blood flow (Fig. 1b). Whereas transmural pressure and circumferential stress in the vessel wall mainly affect and regulate medial VSMCs, fluid shear stress mostly affects endothelial cells, potently regulating their function ${ }^{15-16}$.

## Shear stress on endothelial cells

Shear stress is the frictional force derived from blood viscosity and flow rate that acts tangentially on the endothelial surface ${ }^{15,16,17}$. Due to complex vascular geometries and haemodynamic conditions, shear stress levels and directional patterns vary greatly in different regions

# Review article 

![img-0.jpeg](img-0.jpeg)

Fig. 1 | Atherosclerosis preferentially develops at sites of disturbed flow.
a, Stages of atherosclerotic plaque development: LDL particles infiltrate into the subendothelial space in areas of endothelial dysfunction (1); oxidized LDL promotes inflammation (2); circulating monocytes (3) and vascular smooth muscle cells (VSMCs) from the media (4) migrate towards the region of inflammation; macrophages and VSMCs ingest oxidized LDL particles (5) and, eventually, transform into lipid-laden foam cells (6), contributing to the development of atherosclerotic plaques (7). b, The haemodynamic forces acting on the artery wall are blood pressure, circumferential stress and shear stress. The
three layers of artery wall are the intima (which contains endothelial cells), the media (with VSMCs) and the adventitia (which contains fibroblasts). c, Common sites of atherosclerosis development, with the associated prevalence of plaques in middle-aged adults reported in the AWHS and PESA studies and the shear stress level average and ranges, based on available literature ${ }^{18,19}$. d, Time-averaged shear stress levels in the left carotid bifurcation in a healthy individual show that the lateral wall of the internal carotid, a common site of atherosclerosis development, experiences low and oscillating shear stress from disturbed flow. Panel d adapted from ref. 14, Elsevier.
of the vasculature ${ }^{13,14,16,17}$ (Fig. 1c). In straight, non-branching regions of human arteries, viscous forces of blood flow predominate over inertial forces, leading to stable, unidirectional laminar flow, which delivers high-magnitude ULS ( $\sim 15 \mathrm{dyn} / \mathrm{cm}^{2}$ ) to endothelial cells ${ }^{13,16}$. Conversely, in curved and branching regions, inertial forces become more prominent, leading to disturbed, multidirectional oscillatory flow that delivers low-magnitude OSS (approximately $\pm 4 \mathrm{dyn} / \mathrm{cm}^{2}$ ) to the endothelial cell surface ${ }^{14,17}$ (Fig. 1d). The terms 'stable flow' and 'disturbed flow' are typically used in in vivo studies, whereas most in vitro studies use ULS and OSS to describe the experimental flow conditions to which endothelial cells are exposed. To reduce potential confusion and simplify these interchangeable terms, we use stable flow or disturbed flow in this Review, whenever feasible.

The clinical significance of blood flow is that atherosclerosis preferentially develops in curved or branching vascular regions exposed to disturbed flow conditions in the presence of additional risk factors such as hypercholesterolaemia and diabetes. For example, atherosclerotic plaques form preferentially in the lateral wall of the internal carotid artery at the carotid bifurcation, the lesser curvature of the aortic arch, and the proximal portion of the left anterior descending coronary artery ${ }^{18,19}$.

## Animal models of flow-induced atherosclerosis

Clinical observations strongly suggest a correlation between disturbed flow and sites of atherogenesis, but whether disturbed flow directly causes atherosclerosis remained unknown until it was proven by

# Review article 

experimental studies in animal models. Studies using the partial carotid ligation (PCL) and the shear stress-modifying constrictive carotid cuff mouse models directly demonstrated the effect of disturbed flow, or low flow, on atherosclerosis development ${ }^{20,22}$. In the PCL model, three of the four caudal branches of the left common carotid artery (LCA) are surgically ligated without manipulating the LCA itself. The PCL surgery induces disturbed flow in the LCA with characteristic low-magnitude OSS patterns (Fig. 2a). In Apoe ${ }^{-/-}$mice or C57BL/6 mice overexpressing PCSK9 fed a high-fat diet to induce hypercholesterolaemia, the PCL surgery causes rapid development of atherosclerosis in the entire length
of the LCA within 2-3 weeks ${ }^{20,22,23}$. Importantly, in this mouse model, the contralateral right carotid artery (RCA) continues to be exposed to stable flow and does not develop atherosclerotic plaques, serving as an ideal control in the same animal ${ }^{20,22,23}$ (Fig. 2a). The constrictive carotid cuff model involves the implantation of a shear stress-modifying cast over a portion of the RCA (Fig. 2b), which exposes that portion of the RCA to three different shear stress regimes that translate into atherosclerosis-inducing patterns in hypercholesterolaemic Apoe ${ }^{-/-}$ mice fed a Western diet: low-magnitude, stable flow in the region proximal to the cast, which induces the development of vulnerable
![img-1.jpeg](img-1.jpeg)

Fig. 2 | Models of atherosclerosis induced by disturbed flow. a, Schematic representation of the partial carotid artery ligation mouse model of atherosclerosis (left panel). The external carotid artery (ECA), occipital artery (OA) and internal carotid artery (ICA) are surgically ligated (black lines) to induce disturbed flow in the left carotid artery (LCA), which promotes atherosclerosis development in the LCA in hypercholesterolaemic conditions, such as in Apoe ${ }^{-/-}$mice fed a high-fat diet (central panel) ${ }^{20,143}$ and mice with adeno-associated virus (AAV)-mediated PCSK9 overexpression fed a high-fat diet (right image, middle-right and bottom panels; shown by oil-Red-O staining) ${ }^{22}$. By contrast, even in hypercholesterolaemic conditions, the right carotid artery (RCA), which is exposed to stable flow, does not develop atherosclerosis (right image, middle-left panel). Mice without AAV-PCSK9induced hypercholesterolaemia do not develop atherosclerotic plaques in the RCA or the ligated LCA (right image, top panels). b, Schematic representation of the shear-modifying constrictive cuff model. Implanting a constrictive cuff (white bracket) on the RCA shown in the magnetic resonance imaging angiogram (MRA) exposes endothelial cells to low-magnitude, unidirectional, laminar shear stress (ULS) in the proximal region of the cast, high-magnitude ULS within the cuff
and low-magnitude oscillatory shear stress (OSS) in the distal region of the cuff. In hypercholesterolaemic conditions, such as in Apoe ${ }^{-/-}$mice fed a Western diet for 8 weeks, the low-magnitude OSS induces atherosclerotic plaque (P) development with a large lipid core (black arrows) in the carotid artery, as shown by haematoxylin and eosin staining. The vessel lumen is indicated by an asterisk. c, Schematic representation of a cone-and-plate viscometer. d, Schematic representation of a parallel-plate flow chamber. Endothelial cells are exposed to differential shear stress with the use of a rotating Teflon cone in the cone-and-plate viscometer and with computer-generated hydrostatic pressure in the parallel-plate system. C1 and C2, cuffs; LSA, left subclavian artery; RSA, right subclavian artery;STA, superior thyroid artery. Panel a left drawing adapted from ref. 20, APS; central image adapted from ref. 143, CCU; and right images adapted from ref. 22, Elsevier. Panel b adapted from ref. 24 (Kuhlmann, M. T., Cuhlmann, S., Hoppe, J., Krams, R., Evans, P. C., Strijkers, G. J., Nicolay, K., Hermann, S., Schäfers, M. Implantation of a carotid cuff for triggering shear-stress induced atherosclerosis in mice. J. Vis. Exp. (59), e3308, https://doi.org/10.3791/3308 (2012)). Panel c adapted from ref. 17, Elsevier. Panel d adapted from ref. 33, Elsevier.

plaques; high-magnitude, stable flow within the casted region, with no plaque development; and disturbed flow in the region distal to the cast, which induces the development of stable plaques^{21,24}. This mouse model demonstrates flow-dependent atherosclerosis development within a single carotid artery^{21,24} (Fig. 2b). Both animal models have been used in numerous laboratories worldwide and are extremely valuable tools to study flow-dependent endothelial cell function and atherogenic mechanisms in vivo. One major difference between the models is that in the PCL model, atherosclerotic plaques form throughout the entire length (~1 cm) of the LCA, whereas the contralateral RCA remains healthy and serves as a control. Therefore, this model provides sufficient endothelial samples from the LCA and RCA from the same animal to conduct omics studies, such as single-cell RNA sequencing (scRNA-seq) or bulk RNA sequencing, unlike the cuff model, which provides relatively smaller amounts of endothelial samples^{15}.

In addition to these mouse models of disturbed flow-induced atherosclerosis, the use of zebrafish has emerged as a genetically tractable model to examine early events of atherogenesis^{25,26}. Combined with genetic manipulation approaches to reduce blood flow, zebrafish models provide further evidence that disturbed flow causes atherosclerosis under hypercholesterolaemic conditions^{27--31}. An interesting question is whether disturbed flow-induced atherosclerosis occurs with other risk factors, such as diabetes and hypertension, independently of hypercholesterolaemia.

## In vitro flow models

Numerous in vitro models of shear stress, including the cone-and-plate viscometer, the parallel-plate flow chamber and the microfluidic channel have been developed and have been reviewed previously^{14,32--34}. These in vitro bioreactors expose endothelial cells to various shear conditions and can be used to determine the detailed mechanisms of shear stress-dependent endothelial function in a well-defined biomechanical environment (Fig. 2c,d).

## Blood flow regulates endothelial function

Endothelial cells in vivo are constantly exposed to various haemodynamic factors, especially shear stress associated with blood flow, which potently regulates nearly all facets of endothelial cell function. Blood flow regulates vascular tone; endothelial barrier permeability; angiogenesis; endothelial cell proliferation, death, differentiation, senescence, metabolism, inflammation and morphology; and extracellular matrix remodelling^{13,14,16}. Defining the mechanisms by which stable flow protects endothelial homeostatic function and disturbed flow induces endothelial dysfunction is crucial for understanding the pathogenesis of flow-dependent atherosclerosis and developing novel therapeutic approaches.

Blood flow potently regulates endothelial barrier function. Stable flow protects endothelial barrier permeability, whereas disturbed flow promotes endothelial barrier dysfunction^{13,14,16}. Stable flow regulates endothelial cell permeability by promoting tight junction stability via control of occludin expression and attachment to the actin cytoskeleton, as well as control of adherens junction integrity via phosphorylation and degradation of VE-cadherin^{35--37}. Disturbed flow increases both endothelial cell proliferation and apoptosis via multiple mechanisms, including downregulation of the expression of the tumour suppressor protein p53 and inhibition of the anti-apoptotic kinase AKT^{38--41}. Stable flow induces autophagy through a sirtuin 1-dependent and FOXO1-dependent pathway, providing a cytoprotective mechanism for endothelial cells^{42}. Disturbed flow induces endothelial cell senescence through a p53-dependent and sirtuin 1-dependent mechanism, thereby reducing endothelial cell migration and disrupting arterial repair mechanisms^{43}.

Blood flow also regulates metabolism and redox reactions in endothelial cells. Whereas stable flow reduces endothelial glucose uptake and metabolic activity as well as the expression of genes encoding proteins involved in glycolysis, disturbed flow promotes glycolytic metabolism, causing a markedly different metabolomic profile and increased mitochondrial fission^{44--47}. Disturbed flow also increases the endothelial production of reactive oxygen species (ROS) and oxidative stress^{48--51}. This process is mediated by bone morphogenetic protein 4 (BMP4), which induces increased production of superoxide via NADPH oxidases and endothelial nitric oxide synthase (eNOS) uncoupling-dependent mechanisms^{48--52}. Endothelial ROS generation in response to disturbed flow increases vascular oxidative stress and LDL oxidation in the context of atherosclerosis and hypertension^{51,54--57}.

Disturbed flow potently induces endothelial cell inflammation and transdifferentiation of endothelial cells, which have crucial roles in the initiation and progression of atherosclerosis. Disturbed flow induces endothelial inflammation by increasing the expression of endothelial adhesion molecules (VCAM1, ICAM1 and E-selectin), which mediate monocyte adhesion to endothelial cells^{58}. Activation of nuclear factor-κB (NF-κB) signalling is crucial in disturbed flow-induced inflammation^{58}. In addition, disturbed flow promotes the expression of cytokines and chemokines such as IL-1, IL-6 and CCL5 (refs. 58,59). Disturbed flow can induce the transdifferentiation of endothelial cells to mesenchymal cells (endothelial-to-mesenchymal transition; EndMT) and immune-like cells (endothelial-to-immune cell transition; EndIT)^{71}. The pathophysiological importance of EndMT in disturbed flow-induced atherosclerosis has been demonstrated, but the validation and relevance of EndIT in atherosclerosis remain to be determined^{60}. Increased endothelial cell turnover under disturbed flow conditions also coincides with increased transcription of genes encoding angiogenic factors and with increased neovascularization^{61--63}.

Morphologically, endothelial cells adopt an elongated, fusiform shape and align to the direction of flow under stable flow conditions^{12,64}. By contrast, endothelial cells under disturbed flow or no-flow static conditions adopt a polygonal, ‘cobblestone' shape without uniform alignment^{65,66}. These morphological changes are accompanied by actin cytoskeleton remodelling from a pattern of bands encircling the periphery of the cell to a pattern of thick, central stress fibres aligned in the direction of shear stress^{67,68}. These cytoskeletal changes alter intercellular stress and cellular traction forces, affecting subsequent cellular strain and status^{69,70}. In addition, shear stress induces subcellular structural changes, such as changes in nuclear shape and relocation of the Golgi apparatus towards the upstream flow direction^{71,72}.

## Mechanosensors and mechanotransduction

Endothelial cells transduce flow signals into intracellular changes through the processes of mechanosensing and mechanosignal transduction. Endothelial cells recognize fluid shear stress through mechanosensors located in the apical and basal surfaces of the cell, in cell--cell junctions and intracellularly^{73} (Fig. 3). On the apical surface, the mechanosensors include plasma membrane proteins, such as the cation channels PIEZO1 and P2X purinoreceptor 4, NOTCH1, protein kinases, G protein-coupled receptors and plexin D1, as well as membrane-associated structures, such as caveolae, the glycocalyx and primary cilia^{74--87}. PIEZO1 is an inward-rectifying calcium channel present in many cell types that opens in response to mechanical force^{88}. In endothelial cells, PIEZO1 mediates shear stress magnitude-dependent increases in

# Review article 

![img-2.jpeg](img-2.jpeg)

Fig. 3 | Mechanosensors and mechanosignal transduction pathways in endothelial cells. The apical surface of the endothelial cell contains protein mechanosensors, such as plexin D1, NOTCH1, PIEZO1, P2X4 and G protein-coupled receptors (such as GPR68), as well as mechanosensitive cell structures, such as caveolae, primary cilia and the glycocalyx. Cell-cell junctions contain the mechanosensory complex comprising VE-cadherin, platelet endothelial cell adhesion molecule (PECAM1), vascular endothelial growth factor receptor 2 (VEGFR2) and VEGFR3. The basal surface of endothelial cells contains integrin
mechanosensors. Mechanosignal transduction pathways include the PI3K-AKT pathway, ERK1-ERK2 pathway, YAP-TAZ pathway and the RHO signalling pathway. Many mechanosignal transduction pathways result in activation of transcription factors including Krüppel-like factor 2 (KLF2) and KLF4, nuclear factor-kB (NF-kB) and hypoxia-inducible factor 1 $\alpha$ (HIF1 $\alpha$ ). EndMT, endothelial-to-mesenchymal transition; eNOS, endothelial nitric oxide synthase; FAK, focal adhesion kinase; JAM, junctional adhesion molecule; NRF2, nuclear factor erythroid 2-related factor 2; ROS, reactive oxygen species; SOX13, transcription factor SOX13.
intracellular calcium levels, influencing flow-induced, anti-atherogenic and pro-atherogenic responses, such as cell alignment ${ }^{60,87,89}$. NOTCH1 mediates stable flow-induced cellular alignment, suppression of cell proliferation and maintenance of cell-cell junctional integrity, and protects against atherosclerosis development ${ }^{61}$. These effects have been suggested to be controlled by tension-induced NOTCH1 signalling and modulation of intracellular calcium levels ${ }^{51}$. However, another study showed that shear stress-mediated activation of the NOTCH1 response requires PIEZO1, warranting additional clarification regarding the mechanosensory role of NOTCH1 (ref. 76). G protein-coupled receptors, such as the proton-sensing GPR68, undergo conformational changes in response to flow in endothelial cells, mediating shear-induced calcium influx ${ }^{53,90}$. Plexin D1, potentially in connection with the junctional mechanosensory complex, mediates flow-dependent atherosclerosis development by regulating calcium uptake, phosphorylation of vascular endothelial growth factor receptor 2 (VEGFR2), AKT, eNOS, ERK1 and ERK2, as well as induction of the major flow-sensitive transcription factors, Krüppel-like factor 2 (KLF2) and KLF4 (ref. 82).

In cell-cell junctions, platelet endothelial cell adhesion molecule (PECAM1), VE-cadherin and VEGFR2 form the junctional mechanosensory complex, which mediates integrin activation, cellular alignment and eNOS activation in response to stable flow via PI3K-AKT signalling ${ }^{69}$. On the basal surface of endothelial cells, flow induces conformational activation and expression changes in extracellular matrix-binding integrin mechanosensors ${ }^{91,92}$. Stable flow activates the integrin $\alpha v \beta 3$, causing increased binding to the extracellular matrix and inactivation of downstream RHO signalling ${ }^{51}$. In addition,
the actin cytoskeleton in the cytosol has been shown to serve as a mechanosensory structure ${ }^{93}$.

Flow mechanosensing in endothelial cells activates numerous early-to-intermediate (seconds to minutes) cell signalling pathways that lead to long-term (hours to days) atheroprotective or pro-atherogenic processes. However, it is important to note that both pro-atherogenic disturbed flow and atheroprotective stable flow often activate many of the same early-to-intermediate signalling pathways, especially in in vitro studies. For example, stable flow transiently activates NF-kB signalling without leading to long-term endothelial inflammation, whereas disturbed flow induces persistent NF-kB activation resulting in endothelial inflammation ${ }^{94}$. The mechanisms that distinguish these flow-pattern-dependent activation pathways, especially in vitro, are not well understood and remain crucial knowledge gaps that need to be filled. This uncertainty is due in part to the common experimental strategy of conducting in vitro studies with endothelial cells cultured under no-flow conditions that are suddenly subjected to stable flow or disturbed flow. Under these conditions, early-to-intermediateresponses might overlap with common adaptative changes in response to altered mechanical cues regardless of flow patterns and magnitudes. With this caveat in mind, we review the literature that provides crucial knowledge in understanding flow-dependent endothelial cell mechanosignal transduction pathways.

## PIEZO1

PIEZO1 mediates both atheroprotective and pro-atherogenic flowdependent endothelial cell responses ${ }^{78,86,89,95}$. PIEZO1 mechanosensing of stable flow induces intracellular calcium influx, leading to ATP

release and eNOS activation and the production of the atheroprotective and vasorelaxant nitric oxide from endothelial cells^{90,93}. The PIEZO1-induced effect on ATP is mediated by the P2Y_{2} and G_{αq/11} pathways, which activate AKT, which in turn activates eNOS^{94,97}. PIEZO1 also mediates pro-atherogenic endothelial responses of disturbed flow. Disturbed flow stimulates NF-κB activity and endothelial inflammation via PIEZO1-G_{αq/11}-mediated integrin activation, which in turn activates the focal adhesion kinase (FAK)^{74}. Endothelial cell-targeted deletion of Piezo1 in Ldlr-knockout mice inhibited atherosclerotic plaque development in disturbed flow regions, suggesting a pro-atherogenic role of PIEZO1 in endothelial cells^{74}.

## Plexin D1

Plexin D1 is another mechanosensor that responds to both stable flow and disturbed flow, mediating both atheroprotective and pro-atherogenic responses, respectively. Knockdown of Plxnd1, which encodes plexin D1, in mouse endothelial cells inhibits atheroprotective signal transduction pathways, such as eNOS activation, cell alignment and KLF2 and KLF4 expression in response to stable flow^{82}. Interestingly, Plxnd1 knockout in endothelial cells also prevents pro-atherogenic inflammatory responses, including expression of VCAM1 and CCL2 in response to disturbed flow^{82}. Endothelial cell-specific knockout of Plxnd1 in mice prevented atherosclerosis development in arterial regions with disturbed flow but exacerbated plaque development in arterial regions with stable flow^{82}. These results suggest that plexin D1 is a mechanosensor with dual functions depending on blood flow patterns.

## Junctional mechanosensory complex

Stable flow stimulates eNOS activation through the mechanosensory complex formed by PECAM1, VE-cadherin and VEGFR2 or VEGFR3, which in turn activates the PI3K--AKT pathway^{69}. Stable flow also induces integrin αvβ3 and NF-κB activation through the junctional mechanosensory complex^{69}. However, Pecam1 knockout in mice prevents endothelial inflammation and atherosclerosis in arterial regions with disturbed flow^{89,98}. This finding suggests that PECAM1 is a mechanosensor that mediates pro-atherogenic effects of disturbed flow in vivo.

## Integrins

Integrin activation in response to stable flow inactivates RHO GTPases through RHO-like GTPase signalling^{99}. RHOA inactivation promotes YAP phosphorylation (at Ser127 and Ser381) in the cytoplasm to maintain an atheroprotective endothelial cell phenotype^{100}. These interactions between integrins and RHO GTPases further activate RAC, leading to the assembly of the junctional mechanosensory complexes^{101}. Additionally, the RHO GTPase CDC42 is polarized and activated in an integrin-dependent manner and subsequently regulates the polarity of the microtubule-organizing centre^{102,103}. Under disturbed flow conditions, the cooperation between RGD-binding integrins (including α5β1 and αvβ3 integrins) and fibronectin has been shown to drive pro-inflammatory signal transduction involving the nuclear translocation of NF-κB, YAP and serine/threonine-protein kinase PAK^{104--107}.

## Flow-sensitive transcription factors

Thus far, we have discussed mechanosignal transduction pathways occurring in an early-to-intermediate timescale, mediated by specific mechanosensors in response to flow in endothelial cells. These relatively acute responses lead to the regulation of downstream, long-term responses, including activation of transcription factors and transcriptional co-activators, such as KLF2 and KLF4, NF-κB, hypoxia-inducible factor 1α (HIF1α), YAP, TAZ and SOX13, which regulate gene expression profiles and cell function^{73}. KLF2 and KLF4 are two of the most flow-sensitive master transcription factors regulating the expression of genes that control anti-atherogenic pathways induced by stable flow, including vasodilatation and antithrombotic and anti-inflammatory pathways^{108--112}. KLF2 reduces the expression of pro-atherogenic genes by competing with NF-κB for transcriptional cofactor CBP--p300 and by promoting the translocation of nuclear factor erythroid 2-related factor 2 (refs.^{113--115}). Stable flow increases KLF2 transcription by sequentially activating the members of the MAP kinase family MEKK3, MEK5 and ERK5, which in turn activates the transcription factors MEF2A and MEF2C^{116}. By contrast, disturbed flow inactivates the ERK5 pathway, leading to the inhibition of KLF2 expression^{116}. In addition, KLF2 expression is suppressed by the flow-regulated microRNA (miRNA), miR-92a^{117--123}.

NF-κB is a well-recognized transcription factor that is activated by flow. Nuclear translocation and activation of NF-κB in endothelial cells is increased transiently by stable flow and persistently by disturbed flow^{124,125}. NF-κB target genes include those encoding VCAM1, ICAM1, E-selectin, HIF1α and numerous cytokines, all of which have a crucial role in atherosclerosis^{34,123,126}. HIF1α is a pro-atherogenic transcription factor that is activated by disturbed flow^{127--129}. HIF1α induces the expression of glycolytic enzymes such as HK2, PFKFB3 and PDK1 (ref.^{130}). YAP and TAZ -- which are transcriptional co-activators induced by the Hippo signalling pathway and are involved in organ growth and development as well as various diseases such as cancer and atherosclerosis -- are also regulated by both stable and disturbed flow^{131--133}. Disturbed flow induces YAP and TAZ nuclear translocation and activation, leading to endothelial cell inflammation and cytoskeletal remodelling and atherosclerosis^{131}. A study published in 2022 identified SOX13 as a novel flow-sensitive transcription factor. Disturbed flow represses the expression of SOX13, which in turn leads to a strong induction of proinflammatory cytokine and chemokine production, including CCL5 and CXCL10, resulting in endothelial inflammation^{134}.

## Omics approaches to study endothelial cells

Omics-based analyses have become standard approaches to determine changes in endothelial cells in response to various flow and disease conditions. Unlike traditional reductionist approaches studying one or a few candidate genes or proteins at a time, the astonishing advances in omics technologies and computational bioinformatics have made it possible to determine changes in genes, proteins and metabolites at a genome-wide, epigenome-wide, proteome-wide and metabolome-wide scale, often at a single-cell resolution, and using a small amount of sample. The application of these approaches using in vitro and in vivo models has generated a plethora of datasets of flow-dependent transcriptomic, epigenomic, proteomic and metabolomic profiles in endothelial cells and blood vessels under healthy and disease conditions^{31,135--140}.

Early transcriptomic studies used bulk RNA and miRNA samples from pooled cultured endothelial cells and animal tissues to conduct microarray and RNA sequencing analyses. These studies identified numerous unexpected flow-sensitive genes, miRNAs and long non-coding RNAs (lncRNAs), generating wide-ranging novel hypotheses regarding their various roles in endothelial cell function and atherosclerosis^{34,17,73}. Numerous flow-sensitive genes (including BMP4, DNMT1, KDM4B, KLF2, KLK10, PLPP3, SEMA7A, THBS1, TMX1 and ZBTB46), miRNAs (including miR-95a and miR-712) and lncRNAs (including

# Review article 

MALAT1 and MANTIS) were identified from these bulk-RNA studies and subsequently validated, as reviewed elsewhere ${ }^{14,17,73,141,142}$. The roles of flow-sensitive miRNAs and lncRNAs in endothelial function have been reviewed previously ${ }^{143,144}$. KLK10 (which encodes kallikrein-related peptidase 10) has been identified as one of the most flow-sensitive genes from a gene-array study using the mouse PCL model ${ }^{135}$. KLK10 expression is increased by stable flow, but nearly lost in response to disturbed flow in endothelial cells in vitro, mouse arteries in vivo and human coronary arteries with advanced atherosclerotic plaques ${ }^{143}$. After the KLK10 protein is produced, it is secreted into the circulating blood (or the conditioned medium in in vitro assays) and functions as an anti-inflammatory and permeability-barrier-protective protein ${ }^{143}$. Interestingly, KLK10, which is a member of the KLK serine/ threonine protein kinase family, lacks inherent protease activity, and its anti-inflammatory and permeability-barrier-protective functions are mediated by protease-activated receptor 1 (PAR1)-dependent and PAR2-dependent pathways ${ }^{143}$. Administration of recombinant KLK10 via tail vein injection or ultrasound-guided delivery of a KLK10 expression vector to the carotid endothelium prevented endothelial inflammation and atherosclerosis development in mice ${ }^{143,144}$, demonstrating the proof of principle that flow-sensitive proteins such as KLK10 could be used as novel anti-atherogenic therapeutics.

Flow induces epigenome-wide changes in endothelial cells, as revealed by a DNA methylome study that used reduced representation bisulfite sequencing of mouse genomic bulk DNA samples combined with microarray analysis of bulk RNA samples of mouse carotid arteries after PCL surgery ${ }^{136}$. This DNA methylome study, together with other studies, showed that disturbed flow regulates DNA methylation patterns in endothelial cells via the DNA methyltransferases DNMT1 and DNMT3 (refs. 136,145,146). Further studies showed that genetic deletion or pharmacological inhibition of DNMT1 prevented endothelial inflammation and atherosclerosis development in Apoe ${ }^{-/-}$mice ${ }^{145}$, demonstrating that flow-sensitive epigenomic modifications could be anti-atherogenic therapeutic targets.

Proteomics studies using advanced mass spectrometry have identified numerous flow-sensitive proteins that are differentially expressed or post-translationally modified in endothelial cells in response to flow ${ }^{142,147,148}$. Analyses of secreted media (secretome) of endothelial cells show that disturbed flow alters the levels of hundreds of proteins, including ANGPT2 and endothelin 1 (ref. 148). A study to determine proteome-wide $S$-sulfhydration changes of reactive cysteines ( $S$-sulfhydrome) in endothelial cells in response to pro-atherogenic conditions in vitro and in vivo identified hundreds of flow-sensitive $S$-sulfhydrated proteins ${ }^{149}$, including integrins, which have an important role in the flow-dependent vascular relaxation response. A metabolomics study using plasma samples from Apoe ${ }^{-/-}$mice subjected to PCL surgery showed that disturbed flow induces significant changes in the levels of hundreds of metabolites, including sphingomyelin and the amino acids methionine and phenylalanine ${ }^{46}$. However, the causal effects of flow-dependent changes in metabolites have not been clearly defined in vivo and further investigation is warranted. The targets identified by omics approaches and their roles in endothelial cell dysfunction and in atherosclerosis are summarized in Table 1.

## Disturbed-flow-induced reprogramming of endothelial cells

Early transcriptomic and epigenomic studies used mouse bulk RNA and DNA samples obtained from pooled endothelial cells collected by carotid flushing after PCL surgery ${ }^{135,136,140}$. The findings from these
studies helped to establish flow-dependent changes in transcriptomic and DNA methylation patterns in endothelial cells in a genome-wide and epigenome-wide manner. However, although these results revealed a definitive list of genes with reduced expression in endothelial cells in response to disturbed flow, identifying genes that are increased under disturbed flow conditions in the PCL model has been difficult. The reason for this dilemma is that disturbed flow induces endothelial cell inflammation and accumulation of other cell types, especially immune cells, in the subendothelial layer, thereby causing substantial contamination of the endothelium-enriched luminal-flushing samples. Therefore, it was difficult to discern whether the increased expression of any gene of interest originated from the endothelial cells or from the contaminating immune cells and VSMCs. To address the difficulty in identifying flow-sensitive genes in endothelial cells, our group carried out scRNA-seq and single-cell assay for transposase-accessible chromatin sequencing (scATAC-seq) ${ }^{15}$.
scRNA-seq enables the study of transcriptional changes at a genome-wide scale at single-cell resolution. scRNA-seq quantifies each gene transcript, both unspliced precursor mRNA (pre-mRNA) and mature, spliced mRNA (mRNA), in each cell, providing insights into gene transcript expression profiles and dynamic cellular status. The results show each gene transcript quantity in every cell, while the pre-mRNA and mRNA levels can be used for trajectory inference analysis, such as pseudotime analysis or RNA velocity analysis ${ }^{151}$. scATAC-seq assays reveal genome-wide epigenomic changes in chromatin accessibility at single-cell resolution, providing data on gene cis-regulatory elements including enhancers, nucleosome positions and transcription factor binding sites. Integration of scRNA-seq and scATAC-seq analyses provides additional independent validation and comparison of gene transcript levels and epigenomic regulatory profiles for each gene, adding another layer of confidence to the data analysis ${ }^{152}$.

Our study of single cells obtained from LCAs (exposed to disturbed flow for 2 days or 2 weeks) and RCAs (exposed to stable flow for 2 days or 2 weeks) in the same mice after PCL surgery enabled the identification of differential transcriptomic and epigenomic changes in endothelial cells and other cell types, in a flow-dependent and time-dependent manner ${ }^{15}$. The scRNA-seq and scATAC-seq results independently showed that disturbed flow induced dynamic changes in cell composition in the mouse carotid arteries in a time-dependent manner. The comparison and integration of scRNA-seq and scATACseq data showed remarkable concordance, demonstrating the reproducibility and validity of each dataset ${ }^{15}$. Each individual cell in these datasets was assigned a specific cell type based on the expression of cell type-specific canonical marker genes. The carotid arteries contained eight endothelial cell clusters, four monocyte-macrophage clusters, one cluster of VSMCs, one of fibroblasts, one of dendritic cells and one of T cells, all varying in terms of cell identity and number in a flow-dependent and time-dependent manner ${ }^{15}$. Most interestingly, carotid artery endothelial cells are heterogeneous and dynamic in response to flow. In the RCA exposed to the healthy stable flow condition, four endothelial cell subclusters (E1-E4) were identified and remained unchanged over time. However, in the LCA exposed to the pro-atherogenic disturbed flow condition, most of the healthy endothelial cell subclusters (E2-E4) were nearly lost, whereas new endothelial cell subclusters (E6 and E8) emerged ${ }^{15}$. In addition, few VSMCs were found in the RCA intima, but disturbed flow increased the VSMC numbers in the LCA ${ }^{15}$, as expected. Fibroblasts were found in both the LCA and RCA, with the highest number found in the LCA in the 2-day disturbed flow condition. Although monocyte and macrophage

Table 1|Flow-sensitive genes in endothelial cells

| Effect in atherosclerosis | Name | Shear stress type; expression regulation | Target genes | Effect in endothelial cells | Refs. |
| :--: | :--: | :--: | :--: | :--: | :--: |
| Protein-coding genes |  |  |  |  |  |
| Anti-atherogenic | NOS3 | ULS; $\uparrow$ | - | Nitric oxide production and maintenance of vascular tone | 173 |
|  | KLF2 | ULS; $\uparrow$ | - | Antioxidative, antithrombotic, maintenance of vascular integrity and endothelial cell identity | 113,174,175 |
|  | KLF4 | ULS; $\uparrow$ | - | Antioxidative, antithrombotic, maintenance of vascular integrity and endothelial cell identity | 135,165,176 |
|  | SOD2 | ULS; $\uparrow$ | - | Antioxidative | 177 |
|  | SOD3 | ULS; $\uparrow$ | - | Antioxidative | 177 |
|  | TIMP3 | ULS; $\uparrow$ | - | Decreased activities of metalloproteinases and extracellular matrix degradation | 178,179 |
|  | PLPP3 | ULS; $\uparrow$ | - | Expression regulated by KLF2 and miR-92; anti-inflammatory | 112 |
|  | ZBTB46 | OSS; $\downarrow$ | - | Promotes endothelial cell quiescence | 180 |
|  | BMPR2 | ULS; $\uparrow$ | - | Inhibits oxidative stress and NF- $\kappa B$ activation in endothelial cells | 181 |
|  | NFE2L2 | ULS; $\uparrow$ | - | Antioxidant responsive element | 182 |
|  | SOX13 | ULS; $\uparrow$ | - | Anti-inflammatory | 134 |
| Pro-atherogenic | CCL2 | $\begin{aligned} & \text { ULS; } \downarrow \\ & \text { OSS; } \uparrow \end{aligned}$ | - | Promotes immune cell adhesion to endothelial cells | 183 |
|  | BMP4 | OSS; $\uparrow$ | - | Promotes oxidative stress and inflammatory responses | 51,57,184-191 |
|  | VCAM1 | OSS; $\uparrow$ | - | Promotes immune cell adhesion to endothelial cells | 192 |
|  | ICAM1 | OSS; $\uparrow$ | - | Promotes immune cell adhesion to endothelial cells | 192 |
|  | NFKB | OSS; $\uparrow$ | - | Increases pro-inflammatory and pro-atherogenic gene expression | 193-196 |
|  | NOX1, NOX2 | OSS; $\uparrow$ | - | Increases superoxide generation; pro-atherogenic effects | $\begin{gathered} 49-51,55,57, \\ 197-200 \end{gathered}$ |
|  | NOX4 | OSS; $\uparrow$ or $\downarrow$ | - | Increases $\mathrm{H}_{2} \mathrm{O}_{2}$ production leading to pro-atherogenic or anti-atherogenic effects | 51,199,201-203 |
|  | MMPs | OSS; $\uparrow$ | - | Increased extracellular matrix degradation | 204-207 |
|  | TP53 | OSS; $\uparrow$ | - | Promotes cell cycle | 39,208 |
|  | GADD45 | OSS; $\uparrow$ | - | Promotes cell growth and proliferation | 39 |
|  | CDKN1A | OSS; $\uparrow$ | - | Promotes cell growth and proliferation | 39,209 |
|  | MAPK1, MAPK3 | OSS; $\uparrow$ | - | Promotes cell growth and proliferation | 210-212 |
|  | THBS1 | OSS; $\uparrow$ | - | Arterial stiffening | 23,213 |
|  | SEMA7A | OSS; $\uparrow$ | - | Increased expression of cell adhesion molecules and monocyte adhesion | 214 |
|  | HIF1A | OSS; $\uparrow$ | - | Promotes glycolysis and angiogenesis | 127-129,215 |
|  | P2RX7 | OSS; $\uparrow$ | - | Induces ATP-dependent p38 signalling | 216 |
|  | KDM4B | ULS; $\downarrow$ | - | Induces EndMT | 217 |
|  | YAP1, TAZ | OSS; $\uparrow$ | - | Increases pro-inflammatory gene expression and atherogenesis by activating JNK | 100,218 |
|  | HAND2 | OSS; $\uparrow$ | - | Low shear-induced transcription factor, increasing matrix degradation | 219 |
|  | TXNDC5 | OSS; $\uparrow$ | - | Destabilizes endothelial nitric oxide synthase | 220 |

# Review article 

Table 1 (continued) | Flow-sensitive genes in endothelial cells

| Effect in atherosclerosis | Name | Shear stress type; expression regulation | Target genes | Effect in endothelial cells | Refs. |
| :--: | :--: | :--: | :--: | :--: | :--: |
| MicroRNAs |  |  |  |  |  |
| Anti-atherogenic | miR-10a | ULS; $\uparrow$ | BTRC, MAP3K7 | Anti-inflammatory | 221 |
|  | miR-19a | ULS; $\uparrow$ | CCND1, HBP1, HMGB1 | Inhibits endothelial cell proliferation | 222,223 |
|  | miR-23b | ULS; $\uparrow$ | E2F1, FOXO4 | Inhibits endothelial cell proliferation and EndMT | 224,225 |
|  | miR-27b | ULS; $\uparrow$ | DLL4, FLT1, SEMA6A, SEMA6D, SPRY2, TGFB | Inhibits angiogenesis, endothelial cell differentiation and vessel integrity | 226-229 |
|  | miR-101 | ULS; $\uparrow$ | ABCA1, CUL3, MTOR | Inhibits endothelial cell proliferation; promotes angiogenesis | $230-232$ |
|  | miR-143-mIR145 | ULS; $\uparrow$ | CAMK2D, CFL1, ELK1, KLF4, PHACTR4, SSH2 | Inhibits inflammation and promotes anti-atherogenic phenotypes in vascular smooth muscle cells | 233-236 |
|  | miR-126 | NA | BCL2, CCL2, DLK1, FOXO3, HMGB1, IRS1, LRP6, VCAM1 | Promotes endothelial cell proliferation and vascular protection, and inhibits apoptosis, inflammation and atherosclerosis | 237-243 |
| Pro-atherogenic | miR-92a | OSS; $\uparrow$ | CXCL1, ITGA5, KLF2, KLF4, PLPP3, SIRT1 | Promotes endothelial inflammation and angiogenesis | $\begin{aligned} & 112,170, \\ & 244-246 \end{aligned}$ |
|  | miR-205 | OSS; $\uparrow$ | TIMP3 | Increases endothelial inflammation and permeability | 181 |
|  | miR-663 | OSS; $\uparrow$ | ATF4, ELK1, KLF2, KLF4, MYOCD, SOCS5, VEGF | Promotes endothelial inflammation and proatherogenic vascular smooth muscle cell phenotype switching | 247,248 |
|  | miR-712 | OSS; $\uparrow$ | TIMP3 | Increases endothelial inflammation and permeability | 150,249 |
|  | miR-21 | OSS; $\uparrow$ | BCL2, PPARA, PTEN | Increases endothelial inflammation and apoptosis | 250-255 |
|  | miR-155 | ULS; $\uparrow$ | MYLK, NOS3, SOCS1 | Inhibits endothelial inflammation, migration, and proliferation, leading to atheroprotection | 256-259 |
|  |  | $\uparrow$ in atherosclerotic plaque macrophages | BCL6 | Pro-atherogenic | 260 |
| Long non-coding RNAs |  |  |  |  |  |
| Unknown | MALAT1 | ULS; $\uparrow$ | miR-22-3p | Inhibits endothelial cell proliferation, angiogenesis and migration | 261,262 |
|  | MANTIS | ULS; $\uparrow$ | BRG1 | Promotes angiogenesis and endothelial cell alignment | 263 |
|  | LINC00341 | ULS; $\uparrow$ | VCAM1 | Anti-inflammatory | 264 |
|  | LISPR1 | ULS; $\uparrow$ | S1PR1 | Promotes endothelial cell migration and angiogenesis | 141,265 |
|  | STEEL | ULS; $\downarrow$ | NOS3, KLF2 | Promotes intact vessel formation | 141,266 |

EndMT, endothelial-to-mesenchymal transition; KLF2, Krüppel-like factor 2; NA, not available; NF-кB, nuclear factor-кB; OSS, oscillatory shear stress; ULS, unidirectional laminar shear stress. Data are from ref. 73.
clusters were rare in the RCA, the LCA had dramatically increased numbers of monocytes and macrophages, reflecting substantial proinflammatory and pro-atherogenic vascular changes in response to disturbed flow. Like macrophages, few dendritic cells and T cells were present in the RCA but their numbers were increased by disturbed flow in the LCA ${ }^{13}$.

Differential gene expression and gene ontology analyses showed that disturbed flow induces numerous changes that favour proatherogenic responses in endothelial cells ${ }^{13}$. To understand the potential underlying mechanisms by which disturbed flow induces endothelial cell phenotype changes, eight different endothelial cell clusters were analysed for their differential gene expression and gene ontology ${ }^{13}$. As expected, the prototypical healthy endothelial cell clusters (E2) expressed the highest levels of the two best known mechanosensitive genes, Klf2 and Klf4, and the expression of these genes was significantly lower in endothelial cell clusters exposed to disturbed flow, such as the
prototypical E8 in the LCA. Disturbed flow induced the expression of genes in the E8 cluster of endothelial cells that were also found to be highly expressed in VSMCs (Acta2 and Tagln), fibroblasts (Dcn1) and immune cells (Cd74, H2-Eb1, H2-Aa and H2-Ab1), indicating potential EndMT and acquisition of immune cell-like features by endothelial cells under the disturbed flow condition. Comparison of the differentially expressed genes between the stable flow-exposed E2 and disturbed flow-exposed E8 clusters by Panther gene ontology analysis showed that disturbed flow induces many well-known biological processes associated with pro-atherogenic pathways, including inflammation, EndMT, apoptosis, angiogenesis and endothelial permeability ${ }^{13}$. A pseudotime trajectory analysis and additional differential gene expression and chromatin accessibility analyses confirmed that E8 cells express higher levels of marker genes for EndMT (Acta2, Cnn1, Snai1 and Tagln) and EndIT (C1qa, C1qb, C5ar1 and Tnf) than E2 cells. The evidence for EndMT and EndIT was further validated by immunofluorescence

staining of the key immune cell marker proteins C1QA and LYZ in CDH5 ${ }^{+}$ endothelial cells in mice ${ }^{15}$. In addition, chronic exposure of human aortic endothelial cells to disturbed flow in vitro induced the expression of EndMT markers (SNAI1 and TAGLN) and EndIT markers (C1QC and CSAR1) ${ }^{17}$, demonstrating that disturbed flow can induce endothelial cell reprogramming in cultured aortic endothelial cells in the absence of any other cell type, such as immune cells. These results demonstrate that disturbed flow induces the transition of endothelial cells to proatherogenic phenotypes, characterized by inflammation, EndMT and EndIT (that is, FIRE) ${ }^{15}$ (Fig. 4).

EndMT has a crucial role in endothelial cell dysfunction and atherosclerosis, whereas the role of EndIT has not been defined. Endothelial cells undergoing EndMT exhibit traits of mesenchymal cells, such as fibroblasts and VSMCs, while losing typical endothelial cell characteristics, including the elongated cell morphology and cell-cell junctional integrity ${ }^{10,153,154}$. Mechanistically, the flowsensitive transforming growth factor- $\beta 1$ (TGF $\beta 1$ ) is a well-known regulator of the expression of EndMT-related genes ${ }^{155,156}$. Furthermore, endothelial cells without primary cilia have been shown to prime flow-induced EndMT via the TGF $\beta$-ALK5-SMAD2/SMAD3 axis ${ }^{157}$. By contrast, AMP-activated protein kinase is activated by stable flow and suppresses inflammation via nitric oxide-mediated inhibition of NF- $\kappa B$ signalling ${ }^{158-160}$. Cells undergoing EndMT have an important role in atherogenesis by contributing to neointimal thickening, vascular remodelling, and plaque progression and stability ${ }^{153,161}$. A meta-analysis using 28 microarray datasets obtained from endothelial cells exposed to various stimuli (including shear stress, different coronaviruses, hyperlipidaemia and lipopolysaccharide) supports the EndIT concept ${ }^{162}$. Nevertheless, the EndIT concept requires further validation by endothelial cell lineage-tracing studies ${ }^{163}$. Although the pathophysiological importance of EndIT in atherogenesis remains to be tested, the role of disturbed flow-induced endothelial inflammation in atherogenesis is clearly defined.
![img-3.jpeg](img-3.jpeg)

Fig. 4 | Single-cell RNA sequencing reveals disturbed-flow-induced reprogramming of endothelial cells. a, Disturbed flow stimulates the transition of healthy endothelial cells (ECs) to mesenchymal cells (EndMT; E8) and to an immune cell-like state (EndIT; E8), as determined by a pseudotime trajectory analysis of single-cell RNA sequencing datasets obtained from a mouse model of partial carotid artery ligation. The dots along the trajectory lines represent the status of the cells transitioning towards differentiated

## Therapeutic implications in atherosclerosis

As discussed in the Introduction, the CANTOS trial ${ }^{12}$ demonstrated that targeting a non-lipid pathway, such as an inflammatory pathway, could be an effective anti-atherogenic therapy. We propose that flowsensitive genes, proteins and pathways in endothelial cells that regulate FIRE, such as endothelial inflammation, EndMT and EndIT, could be promising novel anti-atherogenic targets. In support of this notion, our transcriptomics study conducted in the mouse PCL model of atherosclerosis showed that both statins and blood flow regulate the expression of hundreds of genes, and the transcriptional profile changes are remarkably distinct from each other ${ }^{164}$. This result suggests that flow-dependent and cholesterol pathways have different roles in atherosclerosis, highlighting the rationale for targeting flow-sensitive molecules (genes, proteins and signalling molecules) as a complementary therapeutic approach. Two therapeutic strategies are conceivable: stimulating or increasing stable-flow-induced atheroprotective molecules, or inhibiting disturbed flow-induced pro-atherogenic molecules with the use of small molecules, recombinant proteins or gene therapies delivered in a systemic or targeted manner.

Several stable-flow-induced molecules are promising antiatherosclerotic targets. KLF2 and KLF4 account for $>50 \%$ of all stableflow-induced gene transcription and the encoded proteins affect nearly all facets of atheroprotective responses in endothelial cells ${ }^{165}$. Given their dominant importance, numerous strategies to stimulate KLF2 and KLF4 expression have been proposed. Statins are a well-known inducer of KLF2 expression in cultured endothelial cells ${ }^{123}$. However, whether statins also induce KLF2 and KLF4 expression in vivo under flow-conditions has been disputed given the potent effect of flow on the expression of these genes ${ }^{103,164}$. Betulinic acid has also been shown to induce KLF2 expression, as well as expression of its target gene NOS3 (which encodes eNOS), via the upstream ERK5-MEF2C pathway ${ }^{166}$. PIEZO1agonists (Yoda1, Jedi1 or Jedi2) or antagonists (salvianolic acid B) have been shown to modify KLF2 and KLF4 expression ${ }^{167-169}$. However,
![img-4.jpeg](img-4.jpeg)
cell types. b, Disturbed flow induces epigenomic changes, such as chromatin remodelling, and transcriptomic changes that lead to pro-atherogenic gene expression patterns, which in turn induce flow-induced reprogramming of ECs (which we term as FIRE, an emerging concept that collectively refers to EndMT, EndIT and EC inflammation) and, eventually, atherosclerosis development. VSMC, vascular smooth muscle cell. Panel a adapted with permission from ref. 15, Elsevier.

# Review article 

given the dual atheroprotective and pro-atherogenic roles of PIEZO1, drugs targeting this receptor would require safety and specificity studies in order to be used as atherosclerosis therapies. The use of recombinant KLKIO or targeted overexpression of KLKIO as an anti-atherogenic therapy is discussed above.

Inhibition of disturbed-flow-induced molecules is a promising antiatherosclerosis strategy. Pharmacological inhibition of disturbed-flowinduced HIF 1 $\alpha$ using the small-molecule inhibitor PX-478 was shown to reduce atherosclerosis in mice ${ }^{129}$. Inhibition of disturbed-flow-induced miRNAs, including the antagomiRs of miR-92a, miR-205 or miR-712, effectively reduced atherosclerosis development in mice ${ }^{130,131-133}$. The agent 5-aza-2'-deoxycytidine inhibits the disturbed-flow-induced DNMT activity and prevented atherosclerosis in a mouse model ${ }^{134}$. Numerous flow-sensitive genes, proteins and pathways, including NF- $\kappa B$, YAP, TAZ and BMP4, as well as specific inhibitors, drugs and RNA therapeutics, are suitable for further investigation, but research on therapeutic strategies targeting disturbed-flow-induced atherogenesis is scarce. Developing approaches to overcome this limitation is a major research area to be developed.

## Conclusions

In conclusion, shear stress from blood flow potently regulates phenotypic and functional changes in endothelial cells that either prevent or promote atherogenesis. Endothelial cells transduce these biomechanical cues through mechanosensors that mediate various mechanosignal transduction pathways, which in turn regulate transcriptomic and epigenomic changes and cellular functions. The advent of high-throughput omics combined with in vivo and in vitro experimental models have revealed numerous flow-sensitive genes, proteins and pathways that regulate endothelial cell dysfunction and atherosclerosis development. Whereas stable flow induces an atheroprotective endothelial cell phenotype, disturbed flow induces an atherogenic phenotype characterized by alteration of endothelial morphology and barrier function, impairment of endothelial metabolism, redox regulation, proliferation and apoptosis, induction of inflammatory pathways, and transdifferentiation to other cell types, such as EndMT. Additionally, scRNA-seq and scATAC-seq studies in vivo have revealed that disturbed flow not only induces endothelial inflammation and EndMT, but also EndIT, which we define as FIRE (flow-induced reprogramming of endothelial cells). The mechanisms and roles of FIRE in endothelial dysfunction and atherosclerosis are major unanswered questions that could reveal important novel mechanisms underlying atherosclerosis. Moreover, the flow-sensitive molecules regulating FIRE could be novel therapeutic targets.

Published online: 24 May 2023

## References

1. Libby, P., Ridker, P. M. \& Maseri, A. Inflammation and atherosclerosis. Circulation 105, 1135-1143 (2002).
2. Herrington, W., Lacey, B., Sherliker, P., Armitage, J. \& Lewington, S. Epidemiology of atherosclerosis and the potential to reduce the global burden of atherothrombotic disease. Circ. Res. 118, 535-546 (2016).
3. Davignon, J. \& Ganz, P. Role of endothelial dysfunction in atherosclerosis. Circulation 109, 8127-8132 (2004).
4. Bennett, M. R., Sinha, S. \& Owens, G. K. Vascular smooth muscle cells in atherosclerosis. Circ. Res. 118, 692-702 (2016).
5. Libby, P. The changing landscape of atherosclerosis. Nature 592, 524-533 (2021).
6. Libby, P., Ridker, P. M., Hansson, G. K. \& Atherothrombosis, L. T. N. O. Inflammation in atherosclerosis: from pathophysiology to practice. J. Am. Coll. Cardiol. 54, 2129-2138 (2009).
7. Caro, C. G., Fitz-Gerald, J. M. \& Schroter, R. C. Arterial wall shear and distribution of early atheroma in man. Nature 223, 1159-1160 (1969).
8. VanderLaan, P. A., Reardon, C. A. \& Getz, G. S. Site specificity of atherosclerosis: siteselective responses to atherosclerotic modulators. Arterioscler Thromb. Vasc. Biol. 24, 12-22 (2004).
9. Tarbell, J. M. Mass transport in arteries and the localization of atherosclerosis. Annu. Rev. Biomed. Eng. 5, 79-118 (2003).
10. Fang, Y., Wu, D. \& Birukov, K. G. Mechanosensing and mechanoregulation of endothelial cell functions. Compr. Physiol. 9, 873-904 (2019).
11. Gallego-Colon, E., Daum, A. \& Yosefy, C. Statins and PCSK9 inhibitors: a new lipid-lowering therapy. Eur. J. Pharmacol. 878, 173114 (2020).
12. Ridker, P. M. et al. Antiinflammatory therapy with canakinumab for atherosclerotic disease. N. Engl. J. Med. 377, 1119-1131 (2017).
13. Kwak, B. R. et al. Biomechanical factors in atherosclerosis: mechanisms and clinical implications. Eur. Heart J. 35, 3013-3020 (2014).
14. Tarbell, J. M., Shi, Z. D., Dunn, J. \& Jo, H. Fluid mechanics, arterial disease, and gene expression. Annu. Rev. Fluid Mech. 46, 591-614 (2014).
15. Andueza, A. et al. Endothelial reprogramming by disturbed flow revealed by single-cell RNA and chromatin accessibility study. Cell Rep. 33, 108491 (2020).
16. Chiu, J. J. \& Chien, S. Effects of disturbed flow on vascular endothelium: pathophysiological basis and clinical perspectives. Physiol. Rev. 91, 327-387 (2011).
17. Simmons, R. D., Kumar, S. \& Jo, H. The role of endothelial mechanosensitive genes in atherosclerosis and omics approaches. Arch. Biochem. Biophys. 591, 111-131 (2016).
18. Fernandez-Freia, L. et al. Prevalence, vascular distribution, and multiterritorial extent of subclinical atherosclerosis in a middle-aged cohort: the PESA (progression of early subclinical atherosclerosis) study. Circulation 131, 2104-2113 (2015).
19. Laclaustra, M. et al. Femoral and carotid subclinical atherosclerosis association with risk factors and coronary calcium: the AWHS study. J. Am. Coll. Cardiol. 67, 1263-1274 (2016).
20. Nam, D. et al. Partial carotid ligation is a model of acutely induced disturbed flow, leading to rapid endothelial dysfunction and atherosclerosis. Am. J. Physiol. Heart Circ. Physiol. 297, H1535-H1543 (2009).
21. Cheng, C. et al. Atherosclerotic lesion size and vulnerability are determined by patterns of fluid shear stress. Circulation 113, 2744-2753 (2006).
22. Kumar, S., Kang, D. W., Rezvan, A. \& Jo, H. Accelerated atherosclerosis development in C57Bl6 mice by overexpressing AAV-mediated PCSK9 and partial carotid ligation. Lab. Invest. 97, 935-945 (2017).
23. Kim, C. W. et al. Disturbed flow promotes arterial stiffening through thrombospondin-1. Circulation 136, 1217-1232 (2017).
24. Kuhlmann, M. T. et al. Implantation of a carotid cuff for triggering shear-stress induced atherosclerosis in mice. J. Vis. Exp. https://doi.org/10.3781/3308 (2012).
25. Tang, D., Geng, F., Yu, C. \& Zhang, R. Recent application of zebrafish models in atherosclerosis research. Front. Cell Dev. Biol. 9, 643697 (2021).
26. Schlegel, A. Zebrafish models for dyslipidemia and atherosclerosis research. Front. Endocrinol. 7, 159 (2016).
27. Baek, K. I. et al. Vascular injury in the zebrafish tail modulates blood flow and peak wall shear stress to restore embryonic circular network. Front. Cardiovasc. Med. 9, 641101 (2022).
28. Hsu, J. J. et al. Contractile and hemodynamic forces coordinate Notch1b-mediated outflow tract valve formation. JCI Insight 4, e124460 (2019).
29. Lee, J. et al. 4-Dimensional light-modulation of cardiac trabeculation. J. Clin. Investig. 128, 1679-1690 (2016).
30. Lee, J. et al. Spatial and temporal variations in hemodynamic forces initiate cardiac trabeculation. JCI Insight https://doi.org/10.1172/jci.insight.96672 (2018).
31. Baek, K. I. et al. Flow-responsive vascular endothelial growth factor receptor-protein kinase C isoform epsilon signaling mediates glycolytic metabolites for vascular repair. Antioxid. Redox Signal. 28, 31-43 (2018).
32. Dewey, C. F. Jr., Bussolari, S. R., Gimbrone, M. A. Jr. \& Davies, P. F. The dynamic response of vascular endothelial cells to fluid shear stress. J. Biomech. Eng. 103, 177-185 (1981).
33. Lawrence, M. B., McIntire, L. V. \& Eskin, S. G. Effect of flow on polymorphonuclear leukocytes/endothelial cell adhesion. Blood 70, 1284-1290 (1987).
34. Rezvan, A., Ni, C.-W., Alberts-Grill, N. \& Jo, H. Animal, in vitro, and ex vivo models of flowdependent atherosclerosis: role of oxidative stress. Antioxid. Redox Signal. 15, 1433-1448 (2011).
35. Colgan, O. C. et al. Regulation of bovine brain microvascular endothelial tight junction assembly and barrier function by laminar shear stress. Am. J. Physiol. Heart Circ. Physiol. 292, H3190-H3197 (2007).
36. Orsenigo, F. et al. Phosphorylation of VE-cadherin is modulated by haemodynamic forces and contributes to the regulation of vascular permeability in vivo. Nat. Commun. 3, 1208 (2012).
37. Caolo, V. et al. Shear stress and VE-cadherin. Arterioscler. Thromb. Vasc. Biol. 38, 2174-2183 (2018).
38. Levesque, M. J., Nerem, R. M. \& Sprague, E. A. Vascular endothelial cell proliferation in culture and the influence of flow. Biomaterials 11, 702-707 (1990).
39. Lin, X. et al. Molecular mechanism of endothelial growth arrest by laminar shear stress. Proc. Natl Acad. Sci. USA 97, 9385-9389 (2000).
40. Dimmeler, S., Haendeler, J., Rippmann, V., Nehls, M. \& Zeiher, A. M. Shear stress inhibits apoptosis of human endothelial cells. FEBS Lett. 289, 71-74 (1996).
41. Dimmeler, S., Assmus, B., Hermann, C., Haendeler, J. \& Zeiher, A. M. Fluid shear stress stimulates phosphorylation of Akt in human endothelial cells: involvement in suppression of apoptosis. Circ. Res. 83, 334-341 (1998).

# Review article 

42. Liu, J. et al. Shear stress regulates endothelial cell autophagy via redox regulation and Sirt1 expression. Cell Death Dis. 6, e1827 (2015).
43. Warboys, C. M. et al. Disturbed flow promotes endothelial senescence via a p53-dependent pathway. Arterioscler. Thromb. Vasc. Biol. 34, 985-995 (2014).
44. Doddaballapur, A. et al. Laminar shear stress inhibits endothelial cell metabolism via KLF2-mediated repression of PFKFB3. Arterioscler. Thromb. Vasc. Biol. 35, 137-145 (2015).
45. Yamamoto, K., Imamura, H. \& Ando, J. Shear stress augments mitochondrial ATP generation that triggers ATP release and $\mathrm{Ca}^{2+}$ signaling in vascular endothelial cells. Am. J. Physiol. Heart Circ. Physiol. 315, H1477-H1485 (2018).
46. Go, Y. M. et al. Disturbed flow induces systemic changes in metabolites in mouse plasma: a metabolomics study using ApoE ${ }^{-/-}$mice with partial carotid ligation. Am. J. Physiol. Regul. Integr. Comp. Physiol. 308, R62-R72 (2015).
47. Hong, S. G. et al. Flow pattern-dependent mitochondrial dynamics regulates the metabolic profile and inflammatory state of endothelial cells. JCI Insight https://doi.org/ 10.1172/jci.insight. 159286 (2022).
48. Heo, K.-S., Fujiwara, K. \& Abe, J.-I. Disturbed-flow-mediated vascular reactive oxygen species induce endothelial dysfunction. Circ. J. 75, 2722-2730 (2011).
49. Hwang, J. et al. Pulsatile versus oscillatory shear stress regulates NADPH oxidase subunit expression: implication for native LDL oxidation. Circ. Res. 93, 1225-1232 (2003).
50. Hwang, J. et al. Oscillatory shear stress stimulates endothelial production of $\mathrm{O}_{2}^{-}$from $\mathrm{p} 47^{\text {phox }}$-dependent NAD(P)H oxidases, leading to monocyte adhesion. J. Biol. Chem. 278, 41291-41298 (2003).
51. Jo, H., Song, H. \& Mowbray, A. Role of NADPH oxidases in disturbed flow- and BMP4- induced inflammation and atherosclerosis. Antioxid. Redox Signal. 8, 1609-1619 (2006).
52. Zhang, J. X. et al. Low shear stress induces vascular eNOS uncoupling via autophagymediated eNOS phosphorylation. Biochim. Biophys. Acta Mol. Cell. Res. 1865, 709-720 (2018).
53. Chachravilis, M., Zhang, Y. L. \& Frangos, J. A. G protein-coupled receptors sense fluid shear stress in endothelial cells. Proc. Natl Acad. Sci. USA 103, 15463-15468 (2006).
54. Li, L. et al. GTP cyclohydrolase I phosphorylation and interaction with GTP cyclohydrolase feedback regulatory protein provide novel regulation of endothelial tetrahydrobiopterin and nitric oxide. Circ. Res. 106, 328-336 (2010).
55. McNally, J. S. et al. Role of xanthine oxidoreductase and NAD(P)H oxidase in endothelial superoxide production in response to oscillatory shear stress. Am. J. Physiol. Heart Circ. Physiol. 285, H2290-H2297 (2003).
56. Meyer, J. W. \& Schmitt, M. E. A central role for the endothelial NADPH oxidase in atherosclerosis. FEBS Lett. 472, 1-4 (2000).
57. Sorescu, G. P. et al. Bone morphogenic protein 4 produced in endothelial cells by oscillatory shear stress induces monocyte adhesion by stimulating reactive oxygen species production from a nos1-based NADPH oxidase. Circ. Res. 95, 773-779 (2004).
58. Nagel, T., Resnick, N., Dewey, C. F. Jr. \& Gimbrone, M. A. Jr. Vascular endothelial cells respond to spatial gradients in fluid shear stress by enhanced activation of transcription factors. Arterioscler. Thromb. Vasc. Biol. 19, 1825-1834 (1999).
59. Sterpetti, A. V. et al. Shear stress increases the release of interleukin-1 and interleukin-6 by aortic endothelial cells. Surgery 194, 911-914 (1993).
60. Soulihol, C., Harmsen, M. C., Evans, P. C. \& Krenning, G. Endothelial-mesenchymal transition in atherosclerosis. Cardiovasc. Res. 114, 965-977 (2016).
61. Tardy, Y., Resnick, N., Nagel, T., Gimbrone, M. A. Jr. \& Dewey, C. F. Jr. Shear stress gradients remodel endothelial monolayers in vitro via a cell proliferation-migration-loss cycle. Arterioscler. Thromb. Vasc. Biol. 17, 3102-3106 (1997).
62. Tressel, S. L. et al. Angiopoietin-2 stimulates blood flow recovery after femoral artery occlusion by inducing inflammation and arteriogenesis. Arterioscler. Thromb. Vasc. Biol. 28, 1989-1995 (2008).
63. Tressel, S. L., Huang, R. P., Tomsen, N. \& Jo, H. Laminar shear inhibits tubule formation and migration of endothelial cells by an angiopoietin-2 dependent mechanism. Arterioscler. Thromb. Vasc. Biol. 27, 2150-2156 (2007).
64. Nerem, R. M., Levesque, M. J. \& Cornhill, J. F. Vascular endothelial morphology as an indicator of the pattern of blood flow. J. Biomech. Eng. 103, 172-176 (1981).
65. Flaherty, J. T. et al. Endothelial nuclear patterns in the canine arterial tree with particular reference to hemodynamic events. Circ. Res. 30, 23-33 (1972).
66. Langille, B. L. \& Adamson, S. L. Relationship between blood flow direction and endothelial cell orientation at arterial branch sites in rabbits and mice. Circ. Res. 48, 481-488 (1981).
67. Franke, R. P. et al. Induction of human vascular endothelial stress fibres by fluid shear stress. Nature 307, 648-649 (1984).
68. Kim, D. W., Langille, B. L., Wong, M. K. \& Gotlieb, A. I. Patterns of endothelial microfilament distribution in the rabbit aorta in situ. Circ. Res. 64, 21-31 (1989).
69. Tzima, E. et al. A mechanosensory complex that mediates the endothelial cell response to fluid shear stress. Nature 437, 426-431 (2005).
70. Steward, R. Jr., Tambe, D., Hardin, C. C., Krishnan, R. \& Fredberg, J. J. Fluid shear, intercellular stress, and endothelial cell alignment. Am. J. Physiol. Cell Physiol. 308, C657-C664 (2015).
71. Coan, G. E., Wechezak, A. R., Viggers, R. F. \& Sauvage, L. R. Effect of shear stress upon localization of the Golgi apparatus and microtubule organizing center in isolated cultured endothelial cells. J. Cell Sci. 104, 1145-1153 (1993).
72. Kwon, H. B. et al. In vivo modulation of endothelial polarization by Apelin receptor signalling. Nat. Commun. 7, 11805 (2016).
73. Demos, C., Tamargo, I. \& Jo, H. Biomechanical regulation of endothelial function in atherosclerosis. Biomech. Coron. Atheroscler. Plaque 4, 3-47 (2021).
74. Albarrán-Juárez, J. et al. Piezo1 and $\mathrm{O}_{2} / \mathrm{O}_{2}$ promote endothelial inflammation depending on flow pattern and integrin activation. J. Exp. Med. 215, 2655-2672 (2018).
75. Bartosch, A. M. W., Mathews, R. \& Tarbell, J. M. Endothelial glycocalyx-mediated nitric oxide production in response to selective AFM pulling. Biophys. J. 113, 101-108 (2017).
76. Caslo, V. et al. Shear stress activates ADAM10 sheddase to regulate Notch1 via the Piezo1 force sensor in endothelial cells. eLife https://doi.org/10.7554/eLife.50684 (2020).
77. Chuntharpursat-Bon, E. et al. PIEZO1 and PECAM1 interact at cell-cell junctions and partner in endothelial force sensing. Commun. Biol. 6, 358 (2023).
78. Dela Paz, N. G. \& Frangos, J. A. Rapid flow-induced activation of $\mathrm{Ga}_{47}$, is independent of Piezo1 activation. Am. J. Physiol. Cell Physiol. 316, C741-c752 (2019).
79. Florian, J. A. et al. Heparan sulfate proteoglycan is a mechanosensor on endothelial cells. Circ. Res. 93, e136-e142 (2003).
80. Li, J. et al. Piezo1 integration of vascular architecture with physiological force. Nature 515, 279-282 (2014).
81. Mack, J. J. et al. NOTCH1 is a mechanosensor in adult arteries. Nat. Commun. 8, 1620 (2017).
82. Mehta, V. et al. The guidance receptor plexin D1 is a mechanosensor in endothelial cells. Nature 578, 290-295 (2020).
83. Nauli, S. M. et al. Endothelial cilia are fluid shear sensors that regulate calcium signaling and nitric oxide production through polycystin-1. Circulation 117, 1161-1171 (2008).
84. Shin, H. et al. Fine control of endothelial VEGFR-2 activation: caveolae as fluid shear stress shelters for membrane receptors. Biomech. Model Mechanobiol. 18, 5-16 (2019).
85. Wang, S. et al. Endothelial cation channel PIEZO1 controls blood pressure by mediating flow-induced ATP release. J. Clin. Invest. 126, 4527-4536 (2016).
86. Yamamoto, K., Korenaga, R., Kamiya, A. \& Ando, J. Fluid shear stress activates $\mathrm{Ca}^{2+}$ influx into human endothelial cells via P2X4 purinoceptors. Circ. Res. 87, 385-391 (2000).
87. Zheng, Q. et al. Mechanosensitive channel PIEZO1 senses shear force to induce KLF2/4 expression via CaMKII/MEKK3/ERK5 axis in endothelial cells. Cells https://doi.org/ 10.3389/cells11142191 (2022).
88. Ridone, P., Vassalli, M. \& Martinac, B. Piezo1 mechanosensitive channels: what are they and why are they important. Biophys. Rev. 11, 795-805 (2019).
89. Ranaibi, S. S. et al. Piezo1, a mechanically activated ion channel, is required for vascular development in mice. Proc. Natl Acad. Sci. USA 111, 10347-10352 (2014).
90. Xu, J. et al. GPR68 senses flow and is essential for vascular physiology. Cell 173, 762-775 e16 (2018).
91. Tzima, E., del Pozo, M. A., Shattil, S. J., Chien, S. \& Schwartz, M. A. Activation of integrins in endothelial cells by fluid shear stress mediates Rho-dependent cytoskeletal alignment. EMBO J. 20, 4639-4647 (2001).
92. Zhang, C. et al. Coupling of integrin a5 to annexin A2 by flow drives endothelial activation. Circ. Res. 127, 1074-1090 (2030).
93. Schwarz, U. S. \& Gardel, M. L. United we stand: integrating the actin cytoskeleton and cell-matrix adhesions in cellular mechanotransduction. J. Cell Sci. 125, 3051-3060 (2012).
94. Mohan, S., Mohan, N. \& Sprague, E. A. Differential activation of NF-kappa B in human aortic endothelial cells conditioned to specific flow environments. Am. J. Physiol. 273, C572-C578 (1997).
95. Murthy, S. E., Dubin, A. E. \& Patapoutian, A. Piezos thrive under pressure: mechanically activated ion channels in health and disease. Nat. Rev. Mol. Cell Biol. 18, 771-783 (2017).
96. Wang, S. et al. P, Y., and $\mathrm{Gq} / \mathrm{O}_{2}$ control blood pressure by mediating endothelial mechanotransduction. J. Clin. Invest. 125, 3077-3086 (2015).
97. Iring, A. et al. Shear stress-induced endothelial adrenomedullin signaling regulates vascular tone and blood pressure. J. Clin. Invest. 129, 2775-2791 (2019).
98. Harry, B. L. et al. Endothelial cell PECAM-1 promotes atherosclerotic lesions in areas of disturbed flow in ApoE-deficient mice. Arterioscler. Thromb. Vasc. Biol. 28, 2003-2008 (2008).
99. Tzima, E. Role of small GTPases in endothelial cytoskeletal dynamics and the shear stress response. Circ. Res. 98, 176-185 (2006).
100. Wang, L. et al. Integrin-IWf/TAZ-JNK cascade mediates atheroprotective effect of unidirectional shear flow. Nature 540, 579-582 (2016).
101. Civelekoglu-Scholey, G. et al. Model of coupled transient changes of Rac, Rho, adhesions and stress fibres alignment in endothelial cells responding to shear stress. J. Theor. Biol. 232, 569-585 (2005).
102. Tzima, E., Kioisass, W. B., del Pozo, M. A. \& Schwartz, M. A. Localized cdc42 activation, detected using a novel assay, mediates microtubule organizing center positioning in endothelial cells in response to fluid shear stress. J. Biol. Chem. 278, 31020-31023 (2003).
103. Palazzo, A. F. et al. Cdc42, dynein, and dynactin regulate MTOC reorientation independent of Rho-regulated microtubule stabilization. Curr. Biol. 11, 1536-1541 (2001).
104. Orr, A. W. et al. The subendothelial extracellular matrix modulates NF- $\kappa B$ activation by flow: a potential role in atherosclerosis. J. Cell Biol. 169, 191-202 (2005).
105. Chen, J. et al. avβ3 Integrins mediate flow-induced NF- $\kappa B$ activation, proinflammatory gene expression, and early atherogenic inflammation. Am. J. Pathol. 185, 2575-2589 (2015).
106. Stupack, D. G. \& Cheresh, D. A. ECM remodeling regulates angiogenesis: endothelial integrins look for new ligands. Sci. STKE 2002, pe7 (2002).
107. Li, B. et al. c-4ld regulates YAPY3ST phosphorylation to activate endothelial atherogenic responses to disturbed flow. J. Clin. Invest. 129, 1167-1179 (2019).
108. Dekker, R. J. et al. Endothelial KLF2 links local arterial shear stress levels to the expression of vascular tone-regulating genes. Am. J. Pathol. 167, 609-618 (2005).
109. Lin, Z. et al. Kruppel-like factor 2 (KLF2) regulates endothelial thrombotic function. Circ. Res. 96, e48-e57 (2005).

# Review article 

110. van Thienen, J. V. et al. Shear stress sustains atheroprotective endothelial KLF2 expression more potently than statins through mRNA stabilization. Cardiovasc. Res. 72, 231-240 (2006).
111. Young, A. et al. Flow activation of AMP-activated protein kinase in vascular endothelium leads to Kruppel-like factor 2 expression. Arterioscler. Thromb. Vasc. Biol. 29, 1902-1908 (2009).
112. Wu, C. et al. Mechanosensitive PPAR2B regulates endothelial responses to atherorelevant hemodynamic forces. Circ. Res. 117, e41-e53 (2015).
113. SenBanerjee, S. et al. KLF2 is a novel transcriptional regulator of endothelial proinflammatory activation. J. Exp. Med. 199, 1305-1315 (2004).
114. Hsieh, C. Y. et al. Regulation of shear-induced nuclear translocation of the Nrf2 transcription factor in endothelial cells. J. Biomed. Sci. 16, 12 (2009).
115. Fledderus, J. O. et al. KLF2 primes the antioxidant transcription factor Nrf2 for activation in endothelial cells. Arterioscler. Thromb. Vasc. Biol. 28, 1339-1346 (2008).
116. Berk, B. C. Atheroprotective signaling mechanisms activated by steady laminar flow in endothelial cells. Circulation 117, 1082-1089 (2008).
117. Abe, J. \& Berk, B. C. Novel mechanisms of endothelial mechanotransduction. Arterioscler. Thromb. Vasc. Biol. 34, 2378-2386 (2014).
118. Kasler, H. G., Victoria, J., Duramad, O. \& Winoto, A. ERK5 is a novel type of mitogenactivated protein kinase containing a transcriptional activation domain. Mol. Cell Biol. 20, 8382-8389 (2000).
119. Sohn, S. J., Li, D., Lee, L. K. \& Winoto, A. Transcriptional regulation of tissue-specific genes by the ERK5 mitogen-activated protein kinase. Mol. Cell Biol. 25, 8553-8566 (2005).
120. Woo, C. H. et al. ERK5 activation inhibits inflammatory responses via peroxisome proliferator-activated receptor $\delta$ (PPAR $\delta$ ) stimulation. J. Biol. Chem. 281, 32164-32174 (2006).
121. Akaike, M. et al. The hinge-helix 1 region of peroxisome proliferator-activated receptor $\gamma 1$ (PPAR $\gamma 1$ ) mediates interaction with extracellular signal-regulated kinase 5 and PPAR $\gamma 1$ transcriptional activation: involvement in flow-induced PPAR $\gamma$ activation in endothelial cells. Mol. Cell Biol. 24, 8691-8704 (2004).
122. Woo, C. H. et al. Extracellular signal-regulated kinase 5 SUMOylation antagonizes shear stress-induced antiinflammatory response and endothelial nitric oxide synthase expression in endothelial cells. Circ. Res. 102, 538-545 (2008).
123. Parmar, K. M. et al. Statins exert endothelial atheroprotective effects via the KLF2 transcription factor. J. Biol. Chem. 280, 26714-26719 (2005).
124. Nakajima, H. \& Mochizuki, N. Flow pattern-dependent endothelial cell responses through transcriptional regulation. Cell Cycle 16, 1893-1901 (2017).
125. Kempe, S., Nestler, H., Lasse, A. \& Wirth, T. NF- $\kappa \delta$ controls the global pro-inflammatory response in endothelial cells: evidence for the regulation of a pro-atherogenic program. Nucleic Acids Res. 33, 5308-5319 (2005).
126. van Uden, P., Kenneth, N. S. \& Rocha, S. Regulation of hypoxia-inducible factor-1a by NF- $\kappa \delta$. Biochem. J. 412, 477-484 (2008).
127. Feng, S. et al. Mechanical activation of hypoxia-inducible factor 1a drives endothelial dysfunction at atheroprone sites. Arterioscler. Thromb. Vasc. Biol. 37, 2087-2101 (2017).
128. Fernandez Esmerats, J. et al. Disturbed flow increases UBE2C (ubiquitin E2 ligase C) via loss of miR-483-3p, inducing aortic valve calcification by the $\mu \mathrm{VH}_{2}$ (von HippelLindau protein) and HIF-1a (hypoxia-inducible factor-1a) pathway in endothelial cells. Arterioscler. Thromb. Vasc. Biol. 39, 467-481 (2019).
129. Villa-Roel, N. et al. Hypoxia inducible factor 1a inhibitor PX-478 reduces atherosclerosis in mice. Atherosclerosis 344, 30-30 (2022).
130. Wu, D. et al. HIF-1alpha is required for disturbed flow-induced metabolic reprogramming in human and porcine vascular endothelium. eLife https://doi.org/10.7554/el.ife.25217 (2017).
131. Wang, K.-C. et al. Flow-dependent YAP/TAZ activities regulate endothelial phenotypes and atherosclerosis. Proc. Natl Acad. Sci. USA 113, 11525-11530 (2016).
132. Zanconato, F., Cordenonsi, M. \& Piccolo, S. YAP/TAZ at the roots of cancer. Cancer cell 29, 783-803 (2016).
133. Piccolo, S., Dupont, S. \& Cordenonsi, M. The biology of YAP/TAZ: hippo signaling and beyond. Physiol. Rev. 94, 1287-1312 (2014).
134. Demos, C. et al. Sox13 is a novel flow-sensitive transcription factor that prevents inflammation by repressing chemokine expression in endothelial cells. Front. Cardiovasc. Med. 9, 979745 (2022).
135. Ni, C.-W. et al. Discovery of novel mechanosensitive genes in vivo using mouse carotid artery endothelium exposed to disturbed flow. Blood 116, e66-e73 (2010).
136. Dunn, J. et al. Flow-dependent epigenetic DNA methylation regulates endothelial gene expression and atherosclerosis. J. Clin. Invest. 124, 3187-3199 (2014).
137. Chen, X. et al. Plasma metabolomics reveals biomarkers of the atherosclerosis. J. Sep. Sci. 33, 2776-2783 (2010).
138. Goveia, J., Stapor, P. \& Carmeliet, P. Principles of targeting endothelial cell metabolism to treat angiogenesis and endothelial cell dysfunction in disease. EMBO Mol. Med. 6, 1105-1120 (2014).
139. Wu, J. et al. Proteomic identification of endothelial proteins isolated in situ from atherosclerotic aorta via systemic perfusion. J. Proteome Res. 6, 4728-4736 (2007).
140. Ajami, N. E. et al. Systems biology analysis of longitudinal functional response of endothelial cells to shear stress. Proc. Natl Acad. Sci. USA 114, 10990-10995 (2017).
141. Kumar, S., Williams, D., Sur, S., Wang, J.-Y. \& Jo, H. Role of flow-sensitive microRNAs and long noncoding RNAs in vascular dysfunction and atherosclerosis. Vasc. Pharmacol. 114, 76-92 (2019).
142. Simmons, R. D., Kumar, S., Thabet, S. R., Sur, S. \& Jo, H. Omics-based approaches to understand mechanosensitive endothelial biology and atherosclerosis. Wiley Interdiscip. Rev. Syst. Biol. Med. 8, 378-401 (2016).
143. Williams, D. et al. Stable flow-induced expression of KLK10 inhibits endothelial inflammation and atherosclerosis. eLife https://doi.org/10.7554/el.ife.72579 (2022).
144. Liu, R., Qu, S., Xu, Y., Ju, H. \& Dai, Z. Spatial control of robust transgene expression in mouse artery endothelium under ultrasound guidance. Signal Transduct. Target. Ther. 7, 225 (2022).
145. Jiang, Y. Z. et al. Hemodynamic disturbed flow induces differential DNA methylation of endothelial Kruppel-like factor 4 promoter in vitro and in vivo. Circ. Res. 115, 32-43 (2014).
146. Zhou, J., Li, Y.-S., Wang, K.-C. \& Chien, S. Epigenetic mechanism in regulation of endothelial function by disturbed flow: induction of DNA hypermethylation by DNMT1. Cell. Mol. Bioeng. 7, 218-224 (2014).
147. Firasat, S., Hecker, M., Binder, L. \& Asif, A. R. Advances in endothelial shear stress proteomics. Expert. Rev. Proteom. 11, 611-619 (2014).
148. Burghoff, S. \& Schrader, J. R. Secretome of human endothelial cells under shear stress. J. Proteome Res. 10, 1160-1169 (2011).
149. Bibli, S. I. et al. Mapping the endothelial cell S-sulfhydrome highlights the crucial role of integrin sulfhydration in vascular function. Circulation 143, 935-948 (2021).
150. Son, D. J. et al. The atypical mechanosensitive microRNA-712 derived from pre-ribosomal RNA induces endothelial inflammation and atherosclerosis. Nat. Commun. 4, 3000 (2013).
151. Weiler, P., Van den Berge, K., Striest, K. \& Tiberi, S. A guide to trajectory inference and RNA velocity. Methods Mol. Biol. 2564, 269-292 (2023).
152. Stuart, T. et al. Comprehensive integration of single-cell data. Cell 177, 1888-1902.e21 (2019).
153. Chen, P. Y. et al. Endothelial-to-mesenchymal transition drives atherosclerosis progression. J. Clin. Invest. 125, 4514-4528 (2015).
154. Frid, M. O., Kale, V. A. \& Stenmark, K. R. Mature vascular endothelium can give rise to smooth muscle cells via endothelial-mesenchymal transdifferentiation: in vitro analysis. Circ. Res. 90, 1189-1196 (2002).
155. Cooley, B. C. et al. TGF- $\beta$ signaling mediates endothelial-to-mesenchymal transition (EndMT) during vein graft remodeling. Sci. Transl. Med. 6, 227ra234 (2014).
156. Kouzbari, K. et al. Oscillatory shear potentiates latent TGF- $\beta 1$ activation more than steady shear as demonstrated by a novel force generator. Sci. Rep. 9, 6065 (2019).
157. Egorova, A. D. et al. Lack of primary cilia primes shear-induced endothelial-to-mesenchymal transition. Circ. Res. 108, 1093-1101 (2011).
158. Zhang, Y., Qiu, J., Wang, X., Zheng, Y. \& Xia, M. AMP-activated protein kinase suppresses endothelial cell inflammation through phosphorylation of transcriptional coactivator p300. Arterioscler. Thromb. Vasc. Biol. 31, 2897-2908 (2011).
159. Cheng, C. K. et al. Activation of AMPK/mIR-191b axis alleviates endothelial dysfunction and vascular inflammation in diabetic mice. Antioxidants https://doi.org/10.3390/ antiox11061137 (2022).
160. Fisethaler, B., Fleming, J., Keseru, B., Walsh, K. \& Busse, R. Fluid shear stress and NO decrease the activity of the hydroxy-methylglutaryl coenzyme A reductase in endothelial cells via the AMP-activated protein kinase and FoxO1. Circ. Res. 100, e12-e21 (2007).
161. Evrard, S. et al. Endothelial to mesenchymal transition is common in atherosclerotic lesions and is associated with plaque instability. Nat. Commun. 7, 11853 (2016).
162. Shao, Y. et al. Endothelial immunity trained by coronavirus infections, DAMP stimulations and regulation by anti-oxidant NRF2 may contribute to inflammations, myelopoiesis, COVID-19 cytokine storms and thromboembolism. Front. Immunol. 12, 653110 (2021).
163. Plein, A., Fantin, A., Denti, L., Pollard, J. W. \& Ruhrberg, C. Erythro-myeloid progenitors contribute endothelial cells to blood vessels. Nature 562, 223-228 (2018).
164. Kumar, S. et al. Atorvastatin and blood flow regulate expression of distinctive sets of genes in mouse carotid artery endothelium. Curr. Top. Membr. 87, 97-130 (2021).
165. Sangwung, P. et al. KLF2 and KLF4 control endothelial identity and vascular integrity. JCI Insight 2, e91700 (2017).
166. Lee, G. H. et al. Betulinic acid induces eNOS expression via the AMPK-dependent KLF2 signaling pathway. J. Agric. Food Chem. 68, 14523-14530 (2020).
167. Davies, J. E. et al. Using Yoda-1 to mimic laminar flow in vitro: a tool to simplify drug testing. Biochem. Pharmacol. 168, 473-480 (2019).
168. Syeda, R. et al. Chemical activation of the mechanotransduction channel Piezo1. eLife https://doi.org/10.7554/el.ife.07369 (2015).
169. Wang, Y. et al. A lever-like transduction pathway for long-distance chemical- and mechano-gating of the mechanosensitive Piezo1 channel. Nat. Commun. 9, 1300 (2018).
170. Wu, W. et al. Flow-dependent regulation of Krüppel-like factor 2 is mediated by microRNA-92a. Circulation 124, 633-641 (2011).
171. Kumar, S., Kim, C. W., Simmons, R. D. \& Jo, H. Role of flow-sensitive microRNAs in endothelial dysfunction and atherosclerosis: mechanosensitive athero-miRs. Arterioscler. Thromb. Vasc. Biol. 34, 2206-2216 (2014).
172. Meng, X., Yin, J., Yu, X. \& Guo, Y. MicroRNA 205-5p promotes unstable atherosclerotic plaque formation in vivo. Cardiovasc. Drugs Ther. 34, 25-39 (2020).
173. Topper, J. N., Cai, J. X., Fallb, D. \& Gimbrone, M. A. Identification of vascular endothelial genes differentially responsive to fluid mechanical stimuli: cyclooxygenase 2, manganese superoxide dismutase, and endothelial cell nitric oxide synthase are selectively upregulated by steady laminar shear stress. Proc. Natl Acad. Sci. USA 93, 10417-10422 (1998).

# Review article 

174. Dekker, R. J. et al. Prolonged fluid shear stress induces a distinct set of endothelial cell genes, most specifically lung Kruppel-like factor (KLF2). Blood 100, 1689-1698 (2002).
175. Huddlason, J. P., Srinivasan, S., Ahmad, N. \& Lingrel, J. B. Fluid shear stress induces endothelial KLF2 gene expression through a defined promoter region. Biol. Chem. 385, 723-729 (2004).
176. Hamik, A. et al. Kruppel-like factor 4 regulates endothelial inflammation. J. Biol. Chem. 282, 13769-13779 (2007).
177. Topper, J. N. \& Gimbrone, M. A. Jr. Blood flow and vascular gene expression: fluid shear stress as a modulator of endothelial phenotype. Mol. Med. Today 5, 40-46 (1999).
178. Stöhr, R. et al. Loss of TIMP3 exacerbates atherosclerosis in ApoE null mice. Atherosclerosis 235, 438-443 (2014).
179. Son, D. J. et al. Interleukin-32a inhibits endothelial inflammation, vascular smooth muscle cell activation, and atherosclerosis by upregulating Timp3 and Reck through suppressing microRNA-205 biogenesis. Theranostics 7, 2186-2203 (2017).
180. Wang, Y. et al. ZBTB46 is a shear-sensitive transcription factor inhibiting endothelial cell proliferation via gene expression regulation of cell cycle proteins. Lab. Investig. https:// doi.org/10.1038/s41374-016-0060-5 (2018).
181. Kim, C. W. et al. Anti-inflammatory and antiatherogenic role of BMP receptor II in endothelial cells. Arterioscler. Thromb. Vasc. Biol. https://doi.org/10.1181/ATVBAHA.112.300281 (2013).
182. Hosoya, T. et al. Differential responses of the Nrf2-Keap1 system to laminar and oscillatory shear stresses in endothelial cells. J. Biol. Chem. 280, 27244-27250 (2005).
183. Shyy, Y. J., Hsieh, H. J., Usami, S. \& Chien, S. Fluid shear stress induces a biphasic response of human monocyte chemotactic protein 1 gene expression in vascular endothelium. Proc. Natl Acad. Sci. USA 91, 4678-4682 (1994).
184. Miriyala, S. et al. Bone morphogenic protein-4 induces hypertension in mice: role of noggin, vascular NADPH oxidases, and impaired vasorelaxation. Circulation 113, 2818-2825 (2008).
185. Vandroo, A. E., Madamanchi, N. R., Hakim, Z. S., Rojas, M. \& Runge, M. S. Thrombin and NAD(P)H oxidase-mediated regulation of CD44 and BMP4-Id pathway in VSMC, restenosis, and atherosclerosis. Circ. Res. 98, 1254-1263 (2006).
186. Sorescu, G. P. et al. Bone morphogenic protein 4 produced in endothelial cells by oscillatory shear stress stimulates an inflammatory response. J. Biol. Chem. 278, 31128-31135 (2003).
187. Son, J. W. et al. Serum BMP-4 levels in relation to arterial stiffness and carotid atherosclerosis in patients with type 2 diabetes. Biomark. Med. 5, 827-835 (2011).
188. Koga, M. et al. BMP4 is increased in the aortas of diabetic ApoE knockout mice and enhances uptake of oxidized low density lipoprotein into peritoneal macrophages. J. Inflamm. 10, 32 (2013).
189. Janik, M. et al. Platelet bone morphogenetic protein-4 mediates vascular inflammation and neointima formation after arterial injury. Cells https://doi.org/10.3390/cells10082027 (2021).
190. Csaizer, A., Labinskyy, N., Jo, H., Ballabh, P. \& Ungvari, Z. Differential proinflammatory and prooxidant effects of bone morphogenetic protein-4 in coronary and pulmonary arterial endothelial cells. Am. J. Physiol. Heart Circ. Physiol. 295, H569-H577 (2008).
191. Chang, K. et al. Bone morphogenic protein antagonists are coexpressed with bone morphogenic protein 4 in endothelial cells exposed to unstable flow in vitro in mouse aortas and in human coronary arteries: role of bone morphogenic protein antagonists in inflammation and atherosclerosis. Circulation 116, 1258-1266 (2007).
192. Nagel, T., Resnick, N., Atkinson, W. J., Dewey, C. F. Jr. \& Gimbrone, M. A. Jr. Shear stress selectively upregulates intercellular adhesion molecule-1 expression in cultured human vascular endothelial cells. J. Clin. Investig. 94, 885-891 (1994).
193. Pamukcu, B., Lip, G. Y. \& Shantsila, E. The nuclear factor-kappa B pathway in atherosclerosis: a potential therapeutic target for atherothrombotic vascular disease. Thromb. Res. 128, 117-123 (2011).
194. Barnes, P. J. \& Karin, M. Nuclear factor- $\kappa \mathrm{B}$ : a pivotal transcription factor in chronic inflammatory diseases. N. Engl. J. Med. 330, 1066-1071 (1997).
195. van der Heiden, K. et al. Role of nuclear factor $\kappa \mathrm{B}$ in cardiovascular health and disease. Clin. Sci. 118, 593-605 (2010).
196. Hays, L. et al. The NF- $\kappa B$ signal transduction pathway in aortic endothelial cells is primed for activation in regions predisposed to atherosclerotic lesion formation. Proc. Natl Acad. Sci. USA 97, 9052-9057 (2000).
197. De Keulemaer, G. W. et al. Oscillatory and steady laminar shear stress differentially affect human endothelial redox state: role of a superoxide-producing NADH oxidase. Circ. Res. 82, 1094-1101 (1998).
198. Goettsch, C. et al. Arterial flow reduces oxidative stress via an antioxidant response element and Oct-1 binding site within the NADPH oxidase 4 promoter in endothelial cells. Basic Res. Cardiol. 106, 551-561 (2011).
199. Gray, S. P. et al. NADPH oxidase 1 plays a key role in diabetes mellitus-accelerated atherosclerosis. Circulation 127, 1888-1902 (2013).
200. Jeon, H. \& Boo, Y. C. Laminar shear stress enhances endothelial cell survival through a NADPH oxidase 2-dependent mechanism. Biochem. Biophys. Res. Commun. 430, 460-465 (2013).
201. Craige, S. M. et al. Endothelial NADPH oxidase 4 protects ApoE-/- mice from atherosclerotic lesions. Free Radic. Biol. Med. 89, 1-7 (2015).
202. Kim, J., Seo, M., Kim, S. K. \& Bae, Y. S. Flagellin-induced NADPH oxidase 4 activation is involved in atherosclerosis. Sci. Rep. 6, 25437 (2016).
203. Langbein, H. et al. NADPH oxidase 4 protects against development of endothelial dysfunction and atherosclerosis in LDL receptor deficient mice. Eur. Heart J. 37, 1753-1761 (2016).
204. Galis, Z. S., Sukhova, G. K., Lark, M. W. \& Libby, P. Increased expression of matrix metalloproteinases and matrix degrading activity in vulnerable regions of human atherosclerotic plaques. J. Clin. Investig. 94, 2493-2503 (1994).
205. Magid, R., Murphy, T. J. \& Galis, Z. S. Expression of matrix metalloproteinase-9 in endothelial cells is differentially regulated by shear stress. Role of c-Myc. J. Biol. Chem. 278, 32994-32999 (2003).
206. Sho, E. et al. Arterial enlargement in response to high flow requires early expression of matrix metalloproteinases to degrade extracellular matrix. Exp. Mol. Pathol. 73, 142-153 (2002).
207. Yun, S. et al. Transcription factor Sp1 phosphorylation induced by shear stress inhibits membrane type 1-matrix metalloproteinase expression in endothelium. J. Biol. Chem. 277, 34808-34814 (2002).
208. Volger, O. L. et al. Distinctive expression of chemokines and transforming growth factor- $\beta$ signaling in human arterial endothelium during atherosclerosis. Am. J. Pathol. 171, 326-337 (2007).
209. Akimoto, S., Mitsumata, M., Sasaguri, T. \& Yoshida, Y. Laminar shear stress inhibits vascular endothelial cell proliferation by inducing cyclin-dependent kinase inhibitor p21 $2 / 4 / 1 / 2 / 2 / 4 / 4$. Circ. Res. 86, 185-190 (2000).
210. Kadohama, T., Nishimura, K., Hoshino, Y., Sasajima, T. \& Sumpio, B. E. Effects of different types of fluid shear stress on endothelial cell proliferation and survival. J. Cell Physiol. 212, 244-251 (2007).
211. Bao, X., Lu, C. \& Fiengos, J. A. Mechanism of temporal gradients in shear-induced ERK1/2 activation and proliferation in endothelial cells. Am. J. Physiol. Heart Circ. Physiol. 281, H22-H29 (2001).
212. Jo, H. et al. Differential effect of shear stress on extracellular signal-regulated kinase and N-terminal Jun kinase in endothelial cells. G12- and G $\beta / \gamma$-dependent signaling pathways. J. Biol. Chem. 272, 1395-1401 (1997).
213. Moore, R. et al. Thrombospondin-1 deficiency accelerates atherosclerotic plaque maturation in ApoE ${ }^{-/-}$mice. Circ. Res. 103, 1181-1189 (2008).
214. Hu, S. et al. Vascular semaphorin 7A upregulation by disturbed flow promotes atherosclerosis through endothelial $\beta 1$ integrin. Arterioscler. Thromb. Vasc. Biol. 38, 335-343 (2018).
215. Henn, D. et al. MicroRNA-regulated pathways of flow-stimulated angiogenesis and vascular remodeling in vivo. J. Transl. Med. 17, 22 (2019).
216. Green, J. P. et al. Atheroprone flow activates inflammation via endothelial ATP-dependent P2X7-p38 signalling. Cardiovasc. Res. 114, 324-335 (2018).
217. Glaser, S. F. et al. The histone demethylase JMJD2B regulates endothelial-to-mesenchymal transition. Proc. Natl Acad. Sci. USA 117, 4180-4187 (2020).
218. Dupont, S. et al. Role of YAP/TAZ in mechanotransduction. Nature 474, 179-183 (2011).
219. Björck, H. M. et al. Characterization of shear-sensitive genes in the normal rat aorta identifies Hand2 as a major flow-responsive transcription factor. PLoS ONE 7, e52227 (2012).
220. Yeh, C.-F. et al. Targeting mechanosensitive endothelial TXNDC5 to stabilize eNOS and reduce atherosclerosis in vivo. Sci. Adv. 8, eab00096 (2022).
221. Fang, Y., Shi, C., Manduchi, E., Civelek, M. \& Davies, P. F. MicroRNA-10a regulation of proinflammatory phenotype in athero-susceptible endothelium in vivo and in vitro. Proc. Natl Acad. Sci. USA 107, 13450-13455 (2010).
222. Qin, X. et al. MicroRNA-19a mediates the suppressive effect of laminar flow on cyclin D1 expression in human umbilical vein endothelial cells. Proc. Natl Acad. Sci. USA 107, 3240-3244 (2010).
223. Chen, H., Li, X., Liu, S., Gu, L. \& Zhou, X. MircroRNA-19a promotes vascular inflammation and foam cell formation by targeting HBP-1 in atherogenesis. Sci. Rep. 7, 12089 (2017).
224. Wang, K. C. et al. Role of microRNA-23b in flow-regulation of Rb phosphorylation and endothelial cell growth. Proc. Natl Acad. Sci. USA 107, 3234-3239 (2010).
225. Iaconetti, C. et al. Down-regulation of miR-23b induces phenotypic switching of vascular smooth muscle cells in vitro and in vivo. Cardiovasc. Res. 107, 522-533 (2015).
226. Melo, S. A. \& Kalluri, R. Angiogenesis is controlled by miR-27b associated with endothelial lip cells. Blood 119, 2439-2440 (2012).
227. Demolli, S. et al. Shear stress-regulated miR-27b controls pericyte recruitment by repressing SEMA6A and SEMA6D. Cardiovasc. Res. 113, 681-691 (2017).
228. Suzuki, H. I. et al. Regulation of TGF- $\beta$-mediated endothelial-mesenchymal transition by microRNA-27. J. Biochem. 161, 417-420 (2017).
229. Zeng, X., Huang, C., Senavirathna, L., Wang, P. \& Liu, L. miR-27b inhibits fibroblast activation via targeting TGF $\beta$ signaling pathway. BMC Cell Biol. 18, 9 (2017).
230. Chen, K. et al. MicroRNA-101 mediates the suppressive effect of laminar shear stress on mTOR expression in vascular endothelial cells. Biochem. Biophys. Res. Commun. 427, 138-142 (2012).
231. Kim, J. H. et al. Hypoxia-responsive microRNA-101 promotes angiogenesis via heme oxygenase-1/vascular endothelial growth factor axis by targeting cullin 3. Antioxid. Redox Signal. 21, 2469-2482 (2014).
232. Zhang, N. et al. MicroRNA-101 overexpression by IL-6 and TNF- $\alpha$ inhibits cholesterol efflux by suppressing ATP-binding cassette transporter A1 expression. Exp. Cell Res. 336, 33-42 (2015).
233. Cordes, K. R. et al. miR-145 and miR-143 regulate smooth muscle cell fate and plasticity. Nature 460, 705-710 (2009).
234. Kohlstedt, K. et al. AMP-activated protein kinase regulates endothelial cell angiotensinconverting enzyme expression via p53 and the post-transcriptional regulation of microRNA-143/145. Circ. Res. 112, 1150-1158 (2013).

# Review article 

235. Sala, F. et al. MiR-143/145 deficiency attenuates the progression of atherosclerosis in Ldlr-/- mice. Thromb. Haemost. 112, 796-802 (2014).
236. Climent, M. et al. TGF $\beta$ triggers miR-143/145 transfer from smooth muscle cells to endothelial cells, thereby modulating vessel stabilization. Circ. Res. 116, 1753-1764 (2015).
237. Zernecke, A. et al. Delivery of microRNA-126 by apoptotic bodies induces CXCL12-dependent vascular protection. Sci. Signal. 2, rs81 (2009).
238. Zhou, J. et al. Regulation of vascular smooth muscle cell turnover by endothelial cell-secreted microRNA-126: role of shear stress. Circ. Res. 113, 40-51 (2013).
239. Schober, A. et al. MicroRNA-126-5p promotes endothelial proliferation and limits atherosclerosis by suppressing Dlk1. Nat. Med. 20, 368-376 (2014).
240. Wang, Y. et al. MicroRNA-126 attenuates palmitate-induced apoptosis by targeting TRAF7 in HUVECs. Mol. Cell. Biochem. 399, 123-130 (2015).
241. Tang, S. T., Wang, F., Shao, M., Wang, Y. \& Zhu, H. Q. MicroRNA-126 suppresses inflammation in endothelial cells under hyperglycemic condition by targeting HMGB1. Vasc. Pharmacol. 88, 48-55 (2017).
242. Cerutti, C. et al. MiR-126 and miR-126* regulate shear-resistant firm leukocyte adhesion to human brain endothelium. Sci. Rep. 7, 45284 (2017).
243. Tang, F. \& Yang, T. L. MicroRNA-126 alleviates endothelial cells injury in atherosclerosis by restoring autophagic flux via inhibiting of PI3K/Akt/mTOR pathway. Biochem. Biophys. Res. Commun. 495, 1482-1489 (2018).
244. Bonauer, A. et al. MicroRNA-92a controls angiogenesis and functional recovery of ischemic tissues in mice. Science 324, 1710-1713 (2009).
245. Fang, Y. \& Davies, P. F. Site-specific microRNA-92a regulation of Kruppel-like factors 4 and 2 in atherosusceptible endothelium. Arterioscler. Thromb. Vasc. Biol. 32, 979-987 (2012).
246. Loyer, X. et al. Inhibition of microRNA-92a prevents endothelial dysfunction and atherosclerosis in mice. Circ. Res. 114, 434-443 (2014).
247. Ni, C. W., Qiu, H. \& Ju, H. MicroRNA-663 upregulated by oscillatory shear stress plays a role in inflammatory response of endothelial cells. Am. J. Physiol. Heart Circ. Physiol. 300, H1762-H1769 (2011).
248. Afonyushkin, T., Oskolkova, O. V. \& Bochkov, V. N. Permissive role of miR-663 in induction of VEGF and activation of the ATF4 branch of unfolded protein response in endothelial cells by oxidized phospholipids. Atherosclerosis 225, 50-55 (2012).
249. Kheirolomoom, A. et al. Multifunctional nanoparticles facilitate molecular targeting and miRNA delivery to inhibit atherosclerosis in ApoE-/- mice. ACS Nano 9, 8885-8897 (2015).
250. Cheng, Y. \& Zhang, C. MicroRNA-21 in cardiovascular disease. J. Cardiovasc. Transl. Res. 3, 251-255 (2010).
251. Weber, M., Baker, M. B., Moore, J. P. \& Searles, C. D. MiR-21 is induced in endothelial cells by shear stress and modulates apoptosis and eNOS activity. Biochem. Biophys. Res. Commun. 393, 643-648 (2010).
252. Buscaglia, L. E. B. \& Li, Y. Apoptosis and the target genes of microRNA-21. Chin. J. Cancer 30, 371-380 (2011).
253. Zhou, J. et al. MicroRNA-21 targets peroxisome proliferators-activated receptor- $\alpha$ in an autoregulatory loop to modulate flow-induced endothelial inflammation. Proc. Natl Acad. Sci. USA 108, 10355-10360 (2011).
254. McDonald, R. A. et al. miRNA-21 is dysregulated in response to vein grafting in multiple models and genetic ablation in mice attenuates neointima formation. Eur. Heart J. 34, 1636-1643 (2013).
255. Li, S. et al. MicroRNA-21 negatively regulates Treg cells through a TGF- $\beta 1 /$ Smadindependent pathway in patients with coronary heart disease. Cell. Physiol. Biochem. 37, 866-878 (2015).
256. Sun, H. X. et al. Essential role of microRNA-155 in regulating endothelium-dependent vasorelaxation by targeting endothelial nitric oxide synthase. Hypertension 60, 1407-1414 (2012).
257. Weber, M., Kim, S., Patterson, N., Rooney, K. \& Searles, C. D. MiRNA-155 targets myosin light chain kinase and modulates actin cytoskeleton organization in endothelial cells. Am. J. Physiol. Heart Circ. Physiol. 306, H1192-H1203 (2014).
258. He, S., Yang, L., Li, D. \& Li, M. Kruppel-like factor 2-mediated suppression of microRNA-155 reduces the proinflammatory activation of macrophages. PLoS ONE 10, e0139060 (2015).
259. Zhang, H. et al. Genistein protects against ox-LDL-induced inflammation through microRNA-155/SOCS1-mediated repression of NF- $\kappa B$ signaling pathway in HUVECs. Inflammation 40, 1450-1459 (2017).
260. Nazari-Jahantigh, M. et al. MicroRNA-155 promotes atherosclerosis by repressing Bcl6 in macrophages. J. Clin. Invest. 122, 4190-4202 (2012).
261. Michalik, K. M. et al. Long noncoding RNA MALAT1 regulates endothelial cell function and vessel growth. Circ. Res. 114, 1389-1397 (2014).
262. Wang, C., Qu, Y., Suo, R. \& Zhu, Y. Long non-coding RNA MALAT1 regulates angiogenesis following oxygen-glucose deprivation/reoxygenation. J. Cell. Mol. Med. https://doi.org/ 10.1111/jcmm.14204 (2019).
263. Leisegang, M. S. et al. Long noncoding RNA MANTIS facilitates endothelial angiogenic function. Circulation 136, 65-79 (2017).
264. Huang, T. S. et al. LINC00341 exerts an anti-inflammatory effect on endothelial cells by repressing VCAM1. Physiol. Genomics 49, 339-345 (2017).
265. Josipovic, I. et al. Long noncoding RNA LISPR1 is required for S1P signaling and endothelial cell function. J. Mol. Cell. Cardiol. 116, 57-68 (2018).
266. Man, H. S. J. et al. Angiogenic patterning by STEEL, an endothelial-enriched long noncoding RNA. Proc. Natl Acad. Sci. USA 115, 2401-2406 (2018).

## Acknowledgements

I.A.T. is supported by NIH grant SF31HL149285-03. K.I.B. is supported by NIH grants ST32HL007745 and F32HL167625. H.J. is supported by NIH grants HL119798, HL139757 and HL151358. H.J. is also supported by the Wallace H. Coulter Distinguished Faculty Professorship.

## Author contributions

The authors contributed substantially to all aspects of the article.

## Competing interests

H.J. is the founder of Flokines Pharma. The other authors declare no competing interests.

## Additional information

Peer review information Nature Reviews Cardiology thanks Shu Chien and the other, anonymous, reviewer(s) for their contribution to the peer review of this work.

Publisher's note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

Springer Nature or its licensor (e.g. a society or other partner) holds exclusive rights to this article under a publishing agreement with the author(s) or other rightsholder(s): author selfarchiving of the accepted manuscript version of this article is solely governed by the terms of such publishing agreement and applicable law.
(c) Springer Nature Limited 2023