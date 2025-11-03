class ExampleText:
    """
    Example text for entity extraction
    """
    def __init__(self):
        self.text = '''
            Alzheimer's disease (AD) follows a well-characterized temporal progression through
            multiple stages. The preclinical phase begins 15-20 years before symptom onset, marked
            by amyloid-beta accumulation in the brain. This silent phase occurs before any cognitive
            symptoms appear.
            
            After amyloid deposition, tau protein pathology develops. Neurofibrillary tangles form
            5-10 years after initial amyloid accumulation. This temporal sequence is consistent
            across patients: amyloid precedes tau pathology.
            
            Mild Cognitive Impairment (MCI) represents the transitional stage between normal aging
            and dementia. MCI typically emerges 3-5 years before AD diagnosis. During MCI, patients
            experience memory problems but maintain independence in daily activities.
            
            The progression from MCI to mild AD occurs over 2-4 years on average. In early-stage AD,
            patients show significant memory impairment and confusion. This stage lasts 2-4 years.
            
            Moderate AD follows, characterized by increased confusion, language difficulties, and
            behavioral changes. This stage persists for 2-10 years. Finally, severe AD results in
            complete loss of communication ability and total dependence, lasting 1-3 years.
         '''
    def get_text(self) -> str:
        return self.text