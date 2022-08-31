from tkinter import Variable
import ltn

def ComposeClasses(ClassLabel:list):
    import itertools
    return list(itertools.permutations(ClassLabel, 2))




def ExclusionPairedClause(Predicate,ltn_v_x:ltn.Variable,onehot_classes_labels:list,Quantifiers='Forall',connectives='And'):
    Quants = {
        'Forall':ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f"),
        'Exsit':ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e")
    }
    Quant = Quants[Quantifiers]
    Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    And = ltn.Connective(ltn.fuzzy_ops.AndProd())
    def GetSingleClause(ltn_v_x,label1,label2,Quantifiers):
        Clause_str = '{}({}, Not( And(Predicate({},{}), Predicate({},{}) ))'.format(Quantifiers,ltn_v_x.free_vars[0],ltn_v_x.free_vars[0],label1.ConstantName,ltn_v_x.free_vars[0],label2.ConstantName)
        return Quant(ltn_v_x,Not(And(Predicate(ltn_v_x,label1),Predicate(ltn_v_x,label2)))),Clause_str

    AllClauses=[]
    AllClauses_str=[]
    for i,(label_1,label_2) in enumerate(ComposeClasses(onehot_classes_labels)):
        Clause,Clause_str = GetSingleClause(ltn_v_x,label_1,label_2,Quantifiers)
        AllClauses.append(Clause)
        AllClauses_str.append(Clause_str)

    return AllClauses,AllClauses_str


def IsA_PairedClause(Predicate,ltn_v_x_inputs:list,onehot_x_labels:list,Quantifiers='Forall',connectives='And'):
    Quants = {
        'Forall':ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f"),
        'Exsit':ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e")
    }
    Quant = Quants[Quantifiers]
    Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    And = ltn.Connective(ltn.fuzzy_ops.AndProd())
    def GetSingleClause(ltn_v_x,label_x,Quantifiers):
        Clause_str = '{}({}, Predicate({},{}))'.format(Quantifiers,ltn_v_x.free_vars[0],ltn_v_x.free_vars[0],label_x.ConstantName,)
        return Quant(ltn_v_x,Predicate(ltn_v_x,label_x)),Clause_str

    AllClauses=[]
    AllClauses_str=[]
    for i,(ltn_v_x,label_x) in enumerate(zip(ltn_v_x_inputs,onehot_x_labels)):
        Clause,Clause_str = GetSingleClause(ltn_v_x,label_x,Quantifiers)
        AllClauses.append(Clause)
        AllClauses_str.append(Clause_str)

    return AllClauses,AllClauses_str