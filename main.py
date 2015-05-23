import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
import time
import os


np.set_printoptions(threshold=2000)

def logistic(data,label,mode="l2"):
    """docstring for logistic"""
    data = data.fillna(0)
    X_train,X_test,y_train,y_test = train_test_split(data,label,test_size=0.3,random_state=42)

    clf = LogisticRegression(penalty=mode)
    clf.fit(X_train,y_train)
    res = clf.score(X_test,y_test)
    # print res
    # columns = list(data.columns)
    """
    print columns
    print type(columns)
    print len(columns)
    print clf.coef_.tolist()
    print type(clf.coef_.tolist()[0])
    print len(clf.coef_.tolist()[0])
    """
    # out = zip(columns,clf.coef_.tolist()[0])
    # for item in out:
    #     if item[1]>0:
    #         print "\'%s\', "%(item[0]),

    return res

def logistic_less_features(data,label,thresh=0):
    """
    :param data:
    :param label:
    :return: features whose weight larger than thresh
    """
    data = data.fillna(0)
    X_train,X_test,y_train,y_test = train_test_split(data,label,test_size=0.3,random_state=42)

    clf = LogisticRegression(penalty="l1")
    clf.fit(X_train,y_train)
    res = clf.score(X_test,y_test)

    columns = list(data.columns)
    out = zip(columns,clf.coef_.tolist()[0])

    features_left = []

    for item in out:
        if item[1]> thresh:
            features_left.append(item[0])
    return features_left


def logistics_for_features(data,label,thresh=0):

    print "====================================================="
    print "Algorithm:  Logistic Regression"
    time_start = time.time()

    feature_pool = set(data.columns)
    print "##Total number of features: %s" % (len(list(data.columns)))
    feature_selectted = set(logistic_less_features(data,label,thresh=1))
    # print feature_selectted
    feature_left = feature_pool-feature_selectted

    data_selectted = data[list(feature_selectted)]
    # print data_selectted.shape
    print "##Number of selected features: %s" % (len(feature_selectted))
    ac_se = logistic(data_selectted,label)
    print "  Accuracy with selected features: %s" % (ac_se)


    data_left = data[list(feature_left)]
    print data_left.shape

    print "##Number of selected features: %s" % (len(feature_left))
    ac_re = logistic(data_left,label)
    print "  Accuracy without selected features: %s" % (ac_re)

    time_end = time.time()
    print "Time elapse: %s" % (time_end-time_start)

    print "====================================================="





def linear(data,label):
    data = data.fillna(0)
    X_train,X_test,y_train,y_test = train_test_split(data,label,test_size=0.3,random_state=42)

    clf = LinearRegression()
    clf.fit(X_train,y_train)
    res = np.mean(abs(y_test-clf.decision_function(X_test)))
    print res


def rand_forest(data,label):
    """docstring for rand_forest"""
    data = data.fillna(0)
    X_train,X_test,y_train,y_test = train_test_split(data,label,test_size=0.3,random_state=42)

    clf = RFC(max_depth = 10)
    clf.fit(X_train,y_train)
    res = clf.score(X_test,y_test)
    print res


def produce_report(data,label,dataset,algorithm,th,output_file):
    """
    print the report

    :param data:
    :param label:
    :param dataset:  (str) name of dataset
    :param algorithm:  (str) name of algorithm
    :param th: (float) threshhold
    :return: nothing
    """
    output_file.write("\n====================================================="+"\n")
    output_file.write("* Datasets : %s" % (dataset) +"\n")
    output_file.write("* Algorithm: %s" % (algorithm) +"\n")
    output_file.write("* Threshold: %s" % (th) +"\n")

    data = data.fillna(0)
    X_train,X_test,y_train,y_test = train_test_split(data,label,test_size=0.3,random_state=42)

    if algorithm=="Linear Regression":
        clf = LinearRegression()
    elif algorithm=="Logistic Regression_l1":
        clf = LogisticRegression(penalty='l1')
    elif algorithm=="Logistic Regression_l2":
        clf = LogisticRegression(penalty='l2')
    else:
        clf = RFC(max_depth=10)

    time_start = time.time()
    feature_selectted = set(logistic_less_features(data,label,thresh=th))

    feature_pool = set(data.columns)
    output_file.write("* Total number of features: %s" % (len(list(data.columns)))+"\n")
    clf.fit(X_train,y_train)
    if algorithm=="Linear Regression":
        res = np.mean(abs(y_test-clf.decision_function(X_test)))
    else:
        res = clf.score(X_test,y_test)
    output_file.write("  Accuracy of whole features: %s" %(res)+"\n")

    if len(feature_selectted)>0:
        if len(feature_selectted)<10:
            output_file.write("  >>")
            output_file.write(str(feature_selectted))
            output_file.write("\n")
        # print feature_selectted
        feature_left = feature_pool-feature_selectted
        data_selectted = data[list(feature_selectted)]
        output_file.write("* Number of selected features: %s" % (len(feature_selectted))+"\n")
        X_train_s,X_test_s,y_train_s,y_test_s = train_test_split(data_selectted,label,test_size=0.3,random_state=42)
        clf.fit(X_train_s,y_train_s)
        if algorithm=="Linear Regression":
            res_s = np.mean(abs(y_test_s-clf.decision_function(X_test_s)))
        else:
            res_s = clf.score(X_test_s,y_test_s)
        output_file.write("  Accuracy with selected features: %s" % (res_s)+"\n")


        data_left = data[list(feature_left)]
        output_file.write("* Number of features left: %s" % (len(feature_left))+"\n")
        X_train_l,X_test_l,y_train_l,y_test_l = train_test_split(data_left,label,test_size=0.3,random_state=42)
        clf.fit(X_train_l,y_train_l)
        if algorithm=="Linear Regression":
            res_l = np.mean(abs(y_test_l-clf.decision_function(X_test_l)))
        else:
            res_l = clf.score(X_test_l,y_test_l)
        output_file.write("  Accuracy without selected features: %s" % (res_l)+"\n")

    time_end = time.time()
    output_file.write("* Time elapse: %s" % (time_end-time_start)+"\n")

    output_file.write("====================================================="+"\n")
    output_file.write(str(data.shape))
    output_file.write("\n")


def num_of_dummies(df):
    """docstring for num_of_dummies"""
    columns = df.columns
    a = {}
    for i in columns:
        a[i] = len(df[i].unique())
    print a



def main():
    """docstring for main"""
    # df = pd.read_csv("../sample.csv")
    df = pd.read_csv("../D11-02.csv")

    drop_features = [u'idncase', u'idnproceeding',u'comp_date',u'eoirattyid', u'alienattyid',
                     u'flag_mismatch_base_city', u'flag_mismatch_hearing',u'min_osc_date',
                     u'max_osc_date', u'min_input_date', u'max_input_date', u'flag_unknowntime',
                     u'flag_unknownorderwithinday',u'order_raw',u'comp_dow',u'grantraw',u'L1grant2',
                     u'L2grant2',u'lojudgemeanyear', u'lojudgemeannatyear', u'lojudgemeannatdefyear',
                     u'difmeanyear', u'difmeannatyear', u'difmeannatdefyear', u'absdifmeanyear',
                     u'absdifmeannatyear', u'absdifmeannatdefyear', u'outliermeanyear', u'outliermeannatyear',
                     u'outliermeannatdefyear', u'negoutliermeanyear', u'negoutliermeannatyear',u'moderategrantrawnatdef',
                     u'Gender', u'DateofAppointment',u'famcode', u'ij_court_code',u'FirstName', u'LastName',
                     u'FirstUndergrad', u'Judge_name_SLR', u'judge_name_caps', u'OtherLocationsMentioned',
                     u'Year_College_SLR', u'courtid', u'ij_code']
    #  drop famcode and ij_court_code because there are too many of them. drop ij_code, FirstName,LastName,Judge_name_SLR and judge_name_caps
    #  because they are redundant with IJ_NAME; drop courtid because it is redundant with Court_SLR, drop Year_College_SLR
    #  because it is equal to YearofFirstUndergradGraduatio
    label_column = u'grant'
    profile_columns = [u'hearing_loc_code', u'lawyer', u'defensive', u'natid', u'written',
                       u'flag_decisionerror_strdes', u'flag_decisionerror_idncaseproc', u'adj_time_start',
                       u'flag_earlystarttime', u'numinfamily',
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
                       u'JudgeUndergradLocation', u'LawSchool',
                       u'JudgeLawSchoolLocation', u'Bar', u'IJ_NAME',
                       u'Male_judge', u'Court_SLR', u'Year_Appointed_SLR',
                       u'YearofFirstUndergradGraduatio', u'Year_Law_school_SLR',
                       u'President_SLR', u'Government_Years_SLR', u'Govt_nonINS_SLR', u'INS_Years_SLR',
                       u'INS_Every5Years_SLR', u'Military_Years_SLR', u'NGO_Years_SLR', u'Privateprac_Years_SLR',
                       u'Academia_Years_SLR', u'experience',u'log_experience', u'log_gov_experience',
                       u'log_INS_experience', u'log_military_experience', u'log_private_experience',
                       u'log_academic_experience', u'govD', u'INSD', u'militaryD', u'privateD', u'academicD',
                       u'democrat', u'republican',u'hour_start']

    previous_columns = [u'numanycasesperday', u'flag_multiple_proceedings', u'flag_notfirstproceeding',
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
                        u'grantdeny', u'denygrant', u'denydeny']
    df = df.drop(drop_features,axis=1)
    df = df[pd.notnull(df[label_column])]

    # balance the two labels
    df_label_0 = df[df[label_column]==0]
    df_label_1 = df[df[label_column]==1]

    n = min(100,len(df_label_1))
    df = pd.concat([df_label_0.loc[np.random.choice(df_label_0.index, n, replace=False)],
                    df_label_1.loc[np.random.choice(df_label_1.index, n, replace=False)]])


    df_prof = df[profile_columns]
    df_prev = df[previous_columns]
    df_label = df[label_column]

    prof_cate_columns = ['hearing_loc_code','natid',
            'year','hour_start','JudgeUndergradLocation',
            'LawSchool','JudgeLawSchoolLocation','Bar','IJ_NAME',
            'Court_SLR','Year_Appointed_SLR','YearofFirstUndergradGraduatio',
            'Year_Law_school_SLR','President_SLR']

    # <<<<<<<<<<<<<<<<<<<<<<<<< convert categorical features to binary features
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
    df_prof_no_dummy = df_prof.drop(prof_cate_columns,axis=1)
    df_prof = df_prof.drop(prof_cate_columns,axis=1)
    df_prof = df_prof.join(cat_df_prof_after)






    # print ' =============logistics regression=========================== '
    # print "using profile: "
    # logistic(df_prof,df_label)
    #
    # print "\nusing previous data:"
    # logistic(df_prev,df_label)
    #
    # print "\n using all data"
    # whole_df = df_prev.join(df_prof)
    # logistic(whole_df,df_label)


    # print '============ linear regression ==========================='
    #
    # print "using profile: "
    # linear(df_prof,df_label)
    #
    # print "\nusing previous data:"
    # linear(df_prev,df_label)
    #
    # print "\n using all data"
    # whole_df = df_prev.join(df_prof)
    # linear(whole_df,df_label)
    #
    # print '============= random forest ====================='
    #
    #
    # print "using profile: "
    # rand_forest(df_prof,df_label)
    #
    # print "\nusing previous data:"
    # rand_forest(df_prev,df_label)
    #
    # print "\n using all data"
    # whole_df = df_prev.join(df_prof)
    # rand_forest(whole_df,df_label)


    # with or without some features
    # logistics_for_features(df_prof,df_label)
    # logistics_for_features(df_prev,df_label)



    # produce report
    whole_df = df_prev.join(df_prof)

    datasets = [whole_df,df_prev,df_prof,df_prof_no_dummy]
    datasets_name = ["All Feature","Previous Decisions","Profile features","Profile features without categorical features"]
#    Algos = ["Linear Regression","Logistic Regression_l1","Logistic Regression_l2","Random Forest"]
    thresholds = [0,0.5,1,2,5]
    Algos = ["Linear Regression","Logistic Regression_l1","Logistic Regression_l2"]

    try:
        os.mkdir("output")
    except:
        pass

    for i in range(len(datasets)):
        for algo in Algos:
            output_file_name = "output/"+datasets_name[i]+algo+".txt"
            print output_file_name
            output_file = open(output_file_name,'w')
            for th in thresholds:
                produce_report(datasets[i],df_label,datasets_name[i],algo,th,output_file)
            output_file.close()

    # print "grant size: ", df_label.sum()
    # print "sample size: ", len(df_label)




if __name__ == '__main__':
    main()
