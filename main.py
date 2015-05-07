import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cross_validation import train_test_split


def logistic(data,label):
    """docstring for logistic"""
    data = data.fillna(0)
    X_train,X_test,y_train,y_test = train_test_split(data,label,test_size=0.3,random_state=42)
    
    clf = LogisticRegression()
    clf.fit(X_train,y_train)
    res = clf.score(X_test,y_test)
    print res



def main():
    """docstring for main"""
    # df = pd.read_csv("../sample.csv")
    df = pd.read_csv("../D11-02.csv")

    drop_features = ['idncase', 'idnproceeding','comp_date','eoirattyid', 'alienattyid',
                     'flag_mismatch_base_city', 'flag_mismatch_hearing','min_osc_date', 
                     'max_osc_date', 'min_input_date', 'max_input_date', 'flag_unknowntime',
                     'flag_unknownorderwithinday','order_raw','comp_dow','grantraw','L1grant2',
                    'L2grant2',u'lojudgemeanyear', u'lojudgemeannatyear', u'lojudgemeannatdefyear', 
                     u'difmeanyear', u'difmeannatyear', u'difmeannatdefyear', u'absdifmeanyear', 
                     u'absdifmeannatyear', u'absdifmeannatdefyear', u'outliermeanyear', u'outliermeannatyear', 
                     u'outliermeannatdefyear', u'negoutliermeanyear', u'negoutliermeannatyear',u'moderategrantrawnatdef',
                     u'Gender', u'DateofAppointment',u'famcode', ]
    # drop famcode because there are too many of them. 
    label_column = 'grant'
    profile_columns = [u'hearing_loc_code', u'ij_code',u'lawyer', u'defensive', u'natid', u'written', 
                       u'flag_decisionerror_strdes', u'flag_decisionerror_idncaseproc', u'adj_time_start',
                       u'flag_earlystarttime', u'courtid', u'ij_court_code',u'numinfamily',
                       u'numfamsperslot',u'year', u'meangrant_judge', u'numdecisions_judge', u'lomeangrant_judge',
                       u'meangrantraw_judge', u'numdecisionsraw_judge', u'lomeangrantraw_judge', u'moderategrant3070', 
                       u'moderategrantraw3070',u'numdecisions_judgenat', u'lomeangrant_judgenat', u'meangrantraw_judgenat',
                       u'numdecisionsraw_judgenat', u'lomeangrantraw_judgenat', u'meangrant_judgedef', 
                       u'numdecisions_judgedef', u'lomeangrant_judgedef', u'meangrantraw_judgedef', 
                       u'numdecisionsraw_judgedef', u'lomeangrantraw_judgedef', u'meangrant_judgenatdef', 
                       u'numdecisions_judgenatdef', u'lomeangrant_judgenatdef', u'meangrantraw_judgenatdef',
                       u'numdecisionsraw_judgenatdef', u'lomeangrantraw_judgenatdef', u'meangrant_judgelawyer',
                       u'numdecisions_judgelawyer', u'lomeangrant_judgelawyer', u'meangrantraw_judgelawyer', 
                       u'numdecisionsraw_judgelawyer', u'lomeangrantraw_judgelawyer', u'natcourtcode', u'natdefcode',
                       u'natdefcourtcode', u'samenat', u'haseoir', u'samedefensive', u'morning', u'lunchtime', 
                       u'afternoon', u'numcases_judgeday', u'numcases_judge', u'numcases_court', u'numcases_court_hearing',
                       u'avgnumanycasesperday', u'avgnumasylumcasesperday', u'avgnumpeopleperday', u'avgnumfamsperday',
                       u'LastName', u'FirstName', u'FirstUndergrad', u'JudgeUndergradLocation', u'LawSchool', 
                       u'JudgeLawSchoolLocation', u'Bar', u'OtherLocationsMentioned', u'IJ_NAME', u'Judge_name_SLR', 
                       u'Male_judge', u'Court_SLR', u'Year_Appointed_SLR', 
                       u'YearofFirstUndergradGraduatio', u'Year_College_SLR', u'Year_Law_school_SLR', 
                       u'President_SLR', u'Government_Years_SLR', u'Govt_nonINS_SLR', u'INS_Years_SLR', 
                       u'INS_Every5Years_SLR', u'Military_Years_SLR', u'NGO_Years_SLR', u'Privateprac_Years_SLR', 
                       u'Academia_Years_SLR', u'judge_name_caps', u'experience',u'log_experience', u'log_gov_experience',
                       u'log_INS_experience', u'log_military_experience', u'log_private_experience', 
                       u'log_academic_experience', u'govD', u'INSD', u'militaryD', u'privateD', u'academicD',
                       u'democrat', u'republican','hour_start'] 
    
    previous_columns = ['numanycasesperday', u'flag_multiple_proceedings', u'flag_notfirstproceeding', 
                        u'flag_multiple_proceedings2',u'flag_notfirstproceeding2', u'flag_prevprocgrant',
                        u'flag_prevprocdeny', u'numasylumcasesperday',u'numpeopleperday', u'orderwithinday',
                        u'lastindayD', u'L1grant', u'L1grant_sameday', u'L2grant', u'L2grant_sameday',u'numgrant_prev5',
                        u'numgrant_prev6', u'numgrant_prev7', u'numgrant_prev8', u'numgrant_prev9', u'numgrant_prev10',
                        u'prev5_dayslapse', u'prev6_dayslapse', u'prev7_dayslapse', u'prev8_dayslapse', u'prev9_dayslapse',
                        u'prev10_dayslapse', u'raw_order_court', u'numcourtgrant_prev5', u'numcourtgrantself_prev5', 
                        u'numcourtdecideself_prev5', u'numcourtgrant_prev6', u'numcourtdecideself_prev6', 
                        u'numcourtgrantself_prev6', u'numcourtgrant_prev7', u'numcourtdecideself_prev7', 
                        u'numcourtgrantself_prev7', u'numcourtgrant_prev8', u'numcourtdecideself_prev8', 
                        u'numcourtgrantself_prev8', u'numcourtgrant_prev9', u'numcourtdecideself_prev9', 
                        u'numcourtgrantself_prev9', u'numcourtgrant_prev10', u'numcourtdecideself_prev10', 
                        u'numcourtgrantself_prev10', u'numcourtgrant_prev11', u'numcourtdecideself_prev11', 
                        u'numcourtgrantself_prev11', u'numcourtgrant_prev12', u'numcourtdecideself_prev12', 
                        u'numcourtgrantself_prev12', u'numcourtgrant_prev13', u'numcourtdecideself_prev13', 
                        u'numcourtgrantself_prev13', u'numcourtgrant_prev14', u'numcourtdecideself_prev14',
                        u'numcourtgrantself_prev14', u'numcourtgrant_prev15', u'numcourtdecideself_prev15', 
                        u'numcourtgrantself_prev15', u'numcourtgrant_prev16', u'numcourtdecideself_prev16',
                        u'numcourtgrant_prev17', u'numcourtdecideself_prev17', u'numcourtgrantself_prev17', 
                        u'numcourtgrant_prev18', u'numcourtdecideself_prev18', u'numcourtgrantself_prev18', 
                        u'numcourtgrant_prev19', u'numcourtdecideself_prev19', u'numcourtgrantself_prev19', 
                        u'numcourtgrant_prev20', u'numcourtdecideself_prev20', u'numcourtgrantself_prev20', 
                        u'courtprev5_dayslapse', u'courtprev6_dayslapse', u'courtprev7_dayslapse', 
                        u'courtprev8_dayslapse', u'courtprev9_dayslapse', u'courtprev10_dayslapse', 
                        u'courtprev11_dayslapse', u'courtprev12_dayslapse', u'courtprev13_dayslapse', 
                        u'courtprev14_dayslapse', u'courtprev15_dayslapse', u'courtprev16_dayslapse',
                        u'courtprev17_dayslapse', u'courtprev18_dayslapse', u'courtprev19_dayslapse', 
                        u'courtprev20_dayslapse', u'numcourtgrantother_prev5', u'courtprevother5_dayslapse', 
                        u'numcourtgrantother_prev6', u'courtprevother6_dayslapse', u'numcourtgrantother_prev7', 
                        u'courtprevother7_dayslapse', u'numcourtgrantother_prev8', u'courtprevother8_dayslapse', 
                        u'numcourtgrantother_prev9', u'courtprevother9_dayslapse', u'numcourtgrantother_prev10', 
                        u'courtprevother10_dayslapse', u'courtmeanyear', u'courtmeannatyear', 
                        u'courtmeannatdefyear', u'judgemeanyear', u'judgemeannatyear', u'judgemeannatdefyear', 
                        u'judgenumdecyear', u'judgenumdecnatyear', u'judgenumdecnatdefyear',u'grantgrant', 
                        u'grantdeny', u'denygrant', u'denydeny',  ]
    df = df.drop(drop_features,axis=1)
    df = df[pd.notnull(df[label_column])]
    
    df_prof = df[profile_columns]
    df_prev = df[previous_columns]
    df_label = df[label_column]

    prof_cate_columns = ['hearing_loc_code','ij_code','natid','courtid','ij_court_code',
                         'year','hour_start','LastName','FirstName','FirstUndergrad','JudgeUndergradLocation',
                        'LawSchool','JudgeLawSchoolLocation','Bar','OtherLocationsMentioned','IJ_NAME','Judge_name_SLR',
                        'Court_SLR','Year_College_SLR','Year_Appointed_SLR','YearofFirstUndergradGraduatio',
                        'Year_Law_school_SLR','President_SLR','judge_name_caps']    

    cat_df_prof = df_prof[prof_cate_columns]
    cat_dict_prof = cat_df_prof.T.to_dict().values()

    # select the columns which are categorical
    cat_df_prof = df_prof[prof_cate_columns]

    #convert numerical to string 
    cat_df_prof = cat_df_prof.applymap(str)

    # dataframe to dictionary
    cat_dict_prof = cat_df_prof.T.to_dict().values()

    vec = DV(sparse=False)

    #dummy array
    cat_array_prof = vec.fit_transform(cat_dict_prof)

    # convert back to dataframe
    cat_df_prof_after = pd.DataFrame(cat_array_prof)

    # set column name and index
    dummy_columns = vec.get_feature_names()
    cat_df_prof_after.columns = dummy_columns
    cat_df_prof_after.index = df_prof.index

    # replace the categorical columns with the dummy columns
    df_prof = df_prof.drop(prof_cate_columns,axis=1)
    df_prof = df_prof.join(cat_df_prof_after)
    

    print "using profile: "
    logistic(df_prof,df_label)

    print "\nusing previous data:"
    logistic(df_prev,df_label)

    print "\n using all data"
    whole_df = df_prev.join(df_prof)
    logistic(whole_df,df_label)
    
    print "grant size: ",df_label.sum()
    print "sample size: ",len(df_label)

if __name__ == '__main__':
    main()
