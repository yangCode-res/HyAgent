class ExampleText:
    """
    Example text for entity extraction
    """
    def __init__(self):
        self.text  = [
        {
            "id": "PMID_12345678",
            "text": """
            Type 2 diabetes mellitus (T2DM) is a major risk factor for cardiovascular disease (CVD).
            The causal relationship between diabetes and CVD is well-established through multiple pathways.
            
            Hyperglycemia causes endothelial dysfunction by increasing oxidative stress. This directly
            leads to atherosclerosis development. Advanced glycation end products (AGEs) accumulate in
            diabetic patients and cause vascular inflammation, which promotes plaque formation.
            
            Insulin resistance, a hallmark of T2DM, causes dyslipidemia characterized by elevated
            triglycerides and low HDL cholesterol. This lipid abnormality directly contributes to
            atherosclerotic cardiovascular disease (ASCVD).
            
            C-reactive protein (CRP), an inflammation marker, is elevated in T2DM patients. CRP
            regulates vascular smooth muscle cell proliferation and migration, contributing to
            vascular remodeling.
            
            Metformin treats diabetes by improving insulin sensitivity through AMPK activation.
            It may also prevent cardiovascular events through anti-inflammatory effects. SGLT2
            inhibitors prevent heart failure by promoting natriuresis and reducing cardiac preload.
            
            Potential drug targets include: GLP-1 receptor for glucose control and cardioprotection,
            PCSK9 for lipid management, and IL-6 pathway for inflammation reduction.
            """
        },
        {
            "id": "PMID_87654321",
            "text": """
            The molecular mechanisms linking diabetes to cardiovascular complications involve
            multiple interconnected pathways. Chronic hyperglycemia causes mitochondrial
            dysfunction in endothelial cells, leading to increased production of reactive
            oxygen species (ROS). ROS directly damages vascular endothelium and promotes
            inflammatory responses.
            
            Diabetic dyslipidemia is characterized by increased small dense LDL particles that
            are highly atherogenic. These particles penetrate the arterial wall more easily and
            become oxidized, triggering macrophage activation. Macrophage foam cell formation
            is a key step in atherosclerotic plaque development.
            
            The renin-angiotensin-aldosterone system (RAAS) is hyperactivated in diabetes.
            Angiotensin II causes vasoconstriction and promotes cardiac hypertrophy, contributing
            to heart failure development. ACE inhibitors and ARBs prevent these effects and reduce
            cardiovascular mortality in diabetic patients.
            
            Recent evidence suggests that NLRP3 inflammasome activation in diabetes causes
            chronic inflammation. This pathway regulates IL-1β and IL-18 secretion, which
            promote atherosclerosis. Canakinumab, an IL-1β inhibitor, prevents cardiovascular
            events, suggesting this as a therapeutic target.
            """
        }
    ]
    def get_text(self) -> str:
        return self.text

