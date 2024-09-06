#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dash import Dash, dcc, html, Input, Output, dash_table, callback
import pandas as pd
import dash_bootstrap_components as dbc
import dash
import numpy as np
import PythonMeta as PMA
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly
import plotly.graph_objs as go


MainData = pd.read_csv(r"C:\Users\delil\D1\Mydatabase_CSV.csv")
MetaAnalysisData = pd.read_excel(r'C:\Users\delil\D1\MetaAnalysisDataset.xlsx',dtype=str)
MetaData = pd.read_excel(r"C:\Users\delil\D1\AgingMetaboliteDataBaseMetaData.xlsx")
CombineData = pd.read_excel(r"C:\Users\delil\D1\AgingMetaboliteDatabaseCombingData.xlsx")
# must be a CSV or searches with numbers will be output a result

# simplex was used because it had the nice gray background
# this will not work without callback exceptions. I think it is because all pages are ran at the same time
# they are ran at the same time because of the == path because they are all connected


instruction = [html.Br(),html.Br(),html.Br(),'The Aging Metabolite Database is a manually curated database that includes data',
                ' from 100 published papers that focus on metabolites and how the concentration of these metabolites', 
                ' change with age.  It can be used to compare new studies with previously published studies.', 
               html.Br(),html.Br(),
                'The database includes information about the metabolite, such as the metabolite name,', 
                ' InChIKey, KEGG, LIPID MAPS, Pubchem CID, Standardized Name, Formula, Exact Mass, Subclass, and HMDB.',
                html.Br(), html.Br(),
                'It also includes information about the paper, such as the organ, sex, and species tested,', 
                ' if the metabolite increased or decreased with age, and the DOI.',
                html.Br(), html.Br(),
                'Single searches display every instance that the value has been reference while batch searches', 
                ' display how many the times the value has been referenced.',
                html.Br(), html.Br(),
                'Frequency displays how many times the values have been referenced.',
                'This is useful to see what options are available such as what species are in the database.',
                html.Br(), html.Br(),
                'The Meta-analysis tab combines multiple studies together to create a meta-analysis.',
                html.Br(), html.Br(),
                'To create the meta-analysis multiple data point were combined. ',
                'The Raw Meta-analysis Data tab is used to show how these datapoints were combined.',
               html.Br(), html.Br(),
                'The Meta Data tab shows all meta data that was collected for this database.']


batch_placeholder = 'Enter one value per line\nExample:\nblood\nserum\nplasma'



def Search(data,value):
    lists = list(data.split("\n"))
    df = MainData
    # Dash text boxes are all one line in order to make each line a new search item
    # the text has to be split
    a = np.array([])
    b = np.array([])
    c = np.array([])
    d = np.array([])
    e = np.array([])
    f = np.array([])
    g = np.array([])
    h = np.array([])
    j = np.array([])
    k = np.array([])
    l = np.array([])
    m = np.array([])
    n = np.array([])
    o = np.array([])
    p = np.array([])
    q = np.array([])
    for i in lists:
        a = np.append(a,df[df[value].str.contains(i,case=False)==True]['Metabolite Name'].to_numpy())
        b = np.append(b,df[df[value].str.contains(i,case=False)==True]['increased or decreased'].to_numpy())
        c = np.append(c,df[df[value].str.contains(i,case=False)==True]['Organ'].to_numpy())
        d = np.append(d,df[df[value].str.contains(i,case=False)==True]['Sex'].to_numpy())
        e = np.append(e,df[df[value].str.contains(i,case=False)==True]['Age'].to_numpy())
        f = np.append(f,df[df[value].str.contains(i,case=False)==True]['Species'].to_numpy())     
        g = np.append(g,df[df[value].str.contains(i,case=False)==True]['InChIKey'].to_numpy())
        h = np.append(h,df[df[value].str.contains(i,case=False)==True]['Keggs'].to_numpy())
        j = np.append(j,df[df[value].str.contains(i,case=False)==True]['Lipid Maps'].to_numpy())
        k = np.append(k,df[df[value].str.contains(i,case=False)==True]['Standardized Name'].to_numpy())
        l = np.append(l,df[df[value].str.contains(i,case=False)==True]['Formula'].to_numpy())
        m = np.append(m,df[df[value].str.contains(i,case=False)==True]['Exact mass'].to_numpy())
        n = np.append(n,df[df[value].str.contains(i,case=False)==True]['Subclass'].to_numpy())
        o = np.append(o,df[df[value].str.contains(i,case=False)==True]['HMDB'].to_numpy())
        p = np.append(p,df[df[value].str.contains(i,case=False)==True]['DOI'].to_numpy())
        q = np.append(q,df[df[value].str.contains(i,case=False)==True]['Title'].to_numpy())
    columns = {'Metabolite Name': a,
            'Increased or Decreased': b,
            'Organ': c,
            'Sex': d,
            'Age': e,
            'Species': f,
            'InChIKey': g,
            'Keggs': h,
            'Lipid Maps': j,
            'Standardized Name': k,
            'Formula': l,
            'Exact mass': m,
            'Subclass': n,
            'HMDB': o,
            'DOI': p,
            'Title': q }
    #pulls the data thats needed for the datatable given a search
    return pd.DataFrame(columns) 




#Meta Analysis functions

#initialize the metabolite
Metabolite =[
    "Saito2016,1.1151,0.1755,30,0.8613,0.1553,30",
    "Johnson2018,91.42,91.44,29,61.92,40.71,14","Johnson2019,92,94,59,43,21,12"]

#initialize with the first value
SelectedComparison = '100 year olds vs 80 year olds Fecal Acetic acid'

# This is used for the callbacks
# drop down value is the input
# the data connected to the drop down selection is the output
def correctmetabolite(metabolite):
    return MetaAnalysisData.loc[MetaAnalysisData['comparison']==metabolite].reset_index()

def Data(metabolite):
    lists = []
    SelectedMetabolite = correctmetabolite(metabolite)
    for i in range(1,len(SelectedMetabolite)):
        #pulls out the data from the table
        table = SelectedMetabolite.iloc[i].to_numpy()
        # makes a list of only the manditory data 
        SelectedDdata = table[3:]
        lists.append(','.join(SelectedDdata))
    return lists

def Settings(SelectedComparison):
    # when comparing data this asigns what the comparison is
    # "male or age"
    compare1 = correctmetabolite(SelectedComparison).loc[0][4]
    # the other comparison "woman or age"
    compare2 = correctmetabolite(SelectedComparison).loc[0][5]
    #name of metabolite
    MetaboliteName = correctmetabolite(SelectedComparison).loc[0][2]
    # the units of the data
    Units = 'Units: ' + correctmetabolite(SelectedComparison).loc[0][3]
    # what organ this meta analysis is in
    Organ = "Organ: " + correctmetabolite(SelectedComparison).loc[0][6]

    # when the meta analysis has more than 2 studies the random effects model is used
    # when there is only 2 studies the fixed model is used
    if len(Data(SelectedComparison)) > 2:
        Model = 'Random'
    else:
        Model = 'Fixed'     
    
    # when all studies use the same units mean different is used for the effect
    # where there are multiple units used, untargeted unitless studies are used
    # the studies are labeled as unitless and use the standardized mean difference effect size
    if Units == 'Units: Unitless':
        Effect = 'SMD'
    else:
        Effect = 'MD'
        
    return [compare1,compare2,MetaboliteName,Units,Organ,Model,Effect]



def showstudies(studies,dtype,SelectedComparison):
    # this imputs all the data from the previous function
    compare1 = Settings(SelectedComparison)[0]
    compare2 = Settings(SelectedComparison)[1]
    MetaboliteName = Settings(SelectedComparison)[2]
    Units = Settings(SelectedComparison)[3]
    Organ = Settings(SelectedComparison)[4]
    Model = Settings(SelectedComparison)[5]
    Effect = Settings(SelectedComparison)[6]
    #show continuous data
    if dtype.upper()=="CONT":
        allrows = []
        #this pulls out each value, study ID/SD/Mean and put them in their own column
            
        for i in range(len(studies)):
            #allrows.append(new_list)
            allrows.append([
            str(studies[i][6]),        #study ID
            str(studies[i][0]),   #mean of group1
            str(studies[i][1]),   #SD of group1
            str(studies[i][2]),   #total num of group1
            str(studies[i][3]),   #mean of group2
            str(studies[i][4]),   #SD of group2
            str(studies[i][5]) ])   #total num of group2
            
        # this separates the values in the column so that it is a list which can be passed into pandas without changing the format
        studyID = []
        Mean = []
        SD = []
        N = []
        Mean2 = []
        Sd2 = []
        N2 = []
        for i in range(len(allrows)):
            studyID.append(allrows[i][0])
            Mean.append(allrows[i][1])
            SD.append(allrows[i][2])
            N.append(allrows[i][3])
            Mean2.append(allrows[i][4])
            Sd2.append(allrows[i][5])
            N2.append(allrows[i][6])
            

        # this makes a table with only Study ID as the actual title this makes there be no columns
        # the columns are '' ' ' '       ' because the spacing has to be different
        # pandas will think that the columns are the same and stack them if the spacing is the same
        data = {MetaboliteName: [""],Units: [''],Organ: [compare1], '      ':[''],'  ': [' '], '   ':[' '], '    ': [compare2], '     ': ['']}
        
        # makes the dataframe
        df = pd.DataFrame(data)
        # df.loc just adds a row below the data table that is being used as the actual columns names
        df.loc[len(df)] = ['StudyID','Mean', "Standard Deviation", "Sample Size",'----', 'Mean', "Standard Deviation", "Sample Size"]
        # makes a datatable with the actual values being used in the table
        df2data = {MetaboliteName: studyID,Units: Mean,Organ: SD, '      ':N,'  ': '--',
                    '   ':Mean2, '    ': Sd2, '     ': N2}
        df2 = pd.DataFrame(df2data)
        # stack the tables on top of eachother both tables have to have the same column names
        df = pd.concat([df,df2])
        # add a space between this table and the next table
        df.loc[len(df)] = ['','','','','','','','']
        # hides the index       
        df.style.hide_index()
    #return df
    return [studyID,df]



def showresults(rults,SelectedComparison):
    #inputs all the data from the settings like effect size and model used
    compare1 = Settings(SelectedComparison)[0]
    compare2 = Settings(SelectedComparison)[1]
    MetaboliteName = Settings(SelectedComparison)[2]
    Units = Settings(SelectedComparison)[3]
    Organ = Settings(SelectedComparison)[4]
    Model = Settings(SelectedComparison)[5]
    Effect = Settings(SelectedComparison)[6]
    # text 1 is the titles being used
    text1 = "Study ID  N  EffectSize [95%CI]  Weight(%)" 
    StudyID = []
    TotalN = []
    EffectSize = []
    Weight = []
    AllN = []
    totaleffect = []
    LowerConIn = []
    HigherConIn = []
    # this pulls out all of the values as a single string which has to be split below
    for i in range(1,len(rults)):

        StudyID.append(rults[i][0])     #study ID
        TotalN.append(rults[i][5])     #total num
        EF = (str(np.round(rults[i][1],3) ) )   #effect size
        lowerCI = (str(np.round(rults[i][3],3)))     #lower of CI
        LowerConIn.append(np.round(rults[i][3],3))
        higherCI = (str(np.round(rults[i][4],3)) )   #higher of CI 
        EffectSize.append(EF+ "[" + lowerCI + ","+higherCI + ']')                                  
        Weight.append(round(100*(rults[i][2]/rults[0][2]), 2) )#weight
        HigherConIn.append(np.round(rults[i][4],3))

        AllEffect = rults[0][0]     #total effect size name
        AllN = np.round(rults[0][5],3)     #total N (all studies)
        TotalEF = (str(np.round(rults[0][1],3) ) )#total effect size
        LowerCI = (str(np.round(rults[0][3],3))) #total lower CI
        HigherCI = (str(np.round(rults[0][4],3)) )#total higher CI
        totaleffect = (TotalEF+ "[" + LowerCI + ","+ HigherCI + ']')
    weight = np.round(np.sum(Weight),3)


    text3 = "%d studies included  (N=%d)"%(len(rults)-1,rults[0][5])
    text4 = "Heterogeneity:Tau\u00b2=%.3f "%(rults[0][12]) if not rults[0][12]==None else "Heterogeneity:  "
    text4 += " Q(Chisquare)=%.2f  (p=%s);  I\u00b2=%s"%(
        rults[0][7],     #Q test value
        rults[0][8],     #p value for Q test
        str(round(rults[0][9],2))+"%")   #I-square value
    text6 = "Overall effect test:  z=%.2f, p=%s"%(rults[0][10],rults[0][11])  #z-test value and p-value
    
    # makes the tables blank tables with the same columns names as the last table
    data = {MetaboliteName: [""],Units: [''],Organ: [''], '      ':[''],'  ': [' '], 
            '   ':[' '], '    ': [''], '     ': ['']}
    df = pd.DataFrame(data)
    # the actual data table values
    data2 = {MetaboliteName: StudyID,Units: TotalN,Organ: EffectSize, '      ':Weight,'  ': ' ', 
            '   ':' ', '    ': '', '     ': ''}
    
    df2 = pd.DataFrame(data2)
    
    # splits the values so that they are in lists and no a single string
    text1 = text1.split('  ')
    text3 = text3.split('  ')
    text4 = text4.split('  ')
    text6 = text6.split('  ')
    
    # this has all the meta analysis calculations printed
    df.loc[len(df)] = [text1[0],text1[1], text1[2], text1[3],'', "", "", '']
    df = pd.concat([df,df2]) 
    df.loc[len(df)] = ['Total',AllN, totaleffect, weight, '', "", "", '']
    df.loc[len(df)] = [text3[0],text3[1], '', '', '', "", "", '']
    df.loc[len(df)] = [text4[0],text4[1], text4[2], text4[3],'', "", "", '']
    df.loc[len(df)] = [text6[0],text6[1], '', '','', '', "", ""]
    

    return [df,Weight,LowerConIn,HigherConIn,LowerCI,HigherCI]

def thesettings(SelectedComparison):
    # this is how to call the pythonmeta python package
    Model = Settings(SelectedComparison)[5]
    Effect = Settings(SelectedComparison)[6]
    # all data is continuous because the mean and standard deviations are the data points
    settings={
    "datatype":"CONT",  #for CONTinuous data
    "models":Model,             #models: Fixed or Random
    "algorithm":"IV",             #algorithm: IV
    "effect":Effect}
    return settings


def main(stys,settings, SelectedComparison):
    #inputting all of the setting needed for the python meta analysis package
    settings = thesettings(SelectedComparison)
    compare1 = Settings(SelectedComparison)[0]
    compare2 = Settings(SelectedComparison)[1]
    MetaboliteName = Settings(SelectedComparison)[2]
    Units = Settings(SelectedComparison)[3]
    Organ = Settings(SelectedComparison)[4]
    Model = Settings(SelectedComparison)[5]
    Effect = Settings(SelectedComparison)[6]
    
    d = PMA.Data()  #Load Data class
    m = PMA.Meta()  #Load Meta class
    f = PMA.Fig()   #Load Fig class
    

    d.datatype = settings["datatype"]                #set data type, 'CATE' for binary data or 'CONT' for continuous data
    studies = d.getdata(stys)                        #load data
    #get data from a data file, see examples of data files
    showstudies(studies,d.datatype,SelectedComparison)       #show studies

    m.subgroup=d.subgroup

    m.datatype=d.datatype                            #set data type for meta-analysis calculating
    m.models = settings["models"]                    #set effect models: 'Fixed' or 'Random'
    m.algorithm = settings["algorithm"]              #set algorithm, based on datatype and effect size
    m.effect = settings["effect"]                    #set effect size:RR/OR/RD for binary data; SMD/MD for continuous data
    results = m.meta(studies)                        #performing the analysis

    #this is the dataframe made from the showstudies function
    DisplayStudies = showstudies(studies,d.datatype,SelectedComparison)[1]
    #this is just the study IDs
    studyID = showstudies(studies,d.datatype,SelectedComparison)[0]
    # just shows the SMD FIXED IV so the EF type models type and variance calculations
    DisplayStudies.loc[len(DisplayStudies)] = ["Effect measure: "+str(settings["effect"]),"Effect model: "+(str(settings["models"])),"Algorithm: "+str(settings["algorithm"]),'','','','','']
    #this is the dataframe of the results
    MetaAnalysisResults = showresults(results,SelectedComparison)[0]
    #the weights of each study which totals to about 100%
    weight = showresults(results,SelectedComparison)[1]
    # this lower CI of each study
    lowerCI = showresults(results,SelectedComparison)[2]
    # the upper CI of each study
    HigherCI = showresults(results,SelectedComparison)[3]
    # the overall lower CI of the overall total effect
    TotalCILower = showresults(results,SelectedComparison)[4]
    # the upper CI for the total effect
    TotalCIHigher = showresults(results,SelectedComparison)[5]
    #the dataframe that includes the information about the studies (studies)
    # and the results of the meta analysis
    df = pd.concat([DisplayStudies,MetaAnalysisResults])
    # this changes the values to have 2 sigfigs
    def sigfigs(x):
        try:
            return "{:.2f}".format(float(x))
        except:
            return x  

    # this changes the sigfigs of the original table
    DataTable = df.applymap(sigfigs).style.hide_index()
    
    
    # the pythonmeta package outputs f.forest(results) as PythonMeta.core.Fig
    # in order for the figure to be interactive with dash it needs to be a matplotlib figure
    # this changes it to a matplotlib figure so that it can turn into a plotly figure later
    fig = plt.figure(f.forest(results))
    
    # define what is left and what is right "male vs female"

    plt.xlabel("{:<38}{:<s}".format(compare1,compare2),fontsize=17)
    # define titles

    plt.title("Comparing " +compare1 + " and " + compare2,loc="center")
     
    
    return [studyID,fig,DataTable, weight,lowerCI,HigherCI,TotalCILower,TotalCIHigher]



def PlotlyFigure(themetabolite,SelectedComparison):
    #this calls the python meta package
    settings = thesettings(SelectedComparison)
    # this is all the data from the results of the meta analysis
    results = main(themetabolite,settings,SelectedComparison)
    #this is the dataframe of the meta analysis
    meta = results[2].data
    #this pulls the study IDs and adds "overall" as it is the final result of the meta analysis
    # it is the final result
    labels = results[0]+["Overall"]

    tickvals = []
    
    # this is lower CI of each study
    LowerCI = results[4]
    # this is upper CI of each study
    HigherCI = results[5]
    # this is the overall lower CI
    TotalLowerCI = results[6]
    # this is the overall supper CI
    TotalHigherCI = results[7]
    # this is the final effect size value
    totalmidpoint = (float(TotalHigherCI)+float(TotalLowerCI))/2
    # this is the effect size of each study
    midpoint = np.add(HigherCI,LowerCI)/2
    
    #make an array of the labels (StudyID) so that the y axis is labeled with the study ID
    tickvals = []
    for i in range(len(labels)):
        tickvals.append(len(labels)+2-i)

    # this created a space for the overall effect values
    tickvals[-1] = 2

    # this study weights
    study_weights = results[3]

    # converts the matplotlib figure that was created in f.forrest and changed in the precious data
    # it is now a plotly figure
    plotly_figure = mpl_to_plotly(results[1])

    # the hoverdata is the confidence intervals 
    hover_template = "%{x}"
    # this creates the hover data
    plotly_figure.update_traces(hovertemplate=hover_template, selector=dict(type="scatter"),name='Confidence Interval', hoverlabel=dict(font=dict(size=16)))

    # This adds the weights so now the weight of each study is hover data
    plotly_figure.add_trace(go.Scatter(x=midpoint, y=tickvals, hovertemplate='%{marker.size}%',
                                   mode='markers', marker=dict(color='blue', size=study_weights, symbol="square"),name='Weight'))
    
    #updated the figure
    plotly_figure.update_traces(hoverlabel=dict(font=dict(size=16)))
    
    # in the python meta package data the diamond is black with a low I% and white when it is larger
    # matplotlib does not see the black diamond and only the white diamond values
    # to work around the another line is added to the graph which shows a upper CI/ lower CI and effect size
    # it will show up for all calculations so when the diamond is not there you can still see overall effect
    plotly_figure.add_trace(go.Scatter(x=[float(TotalLowerCI), totalmidpoint, float(TotalHigherCI)], 
                                       y=[tickvals[-1],tickvals[-1],tickvals[-1]], 
                                       hovertemplate='%{x}',
                                   mode='lines+markers', marker=dict(color='black', size=3, symbol="square"),name='Confidence Interval'))
    
    
    plotly_figure.update_traces(hoverlabel=dict(font=dict(size=16)))

    # this just shows the sizes of fonts and specifies figure outline
    return [meta,plotly_figure.update_layout(
    width=630, 
    height=680,  title_font_size=19, title_x=0.55,yaxis=dict(tickvals=tickvals, ticktext=labels, tickfont=dict(size=14)),
        xaxis=dict(
        tickfont=dict(size=14)))] 
    
# this initalizes the figure
meta = PlotlyFigure(Metabolite,SelectedComparison)[0]

#the dashboard

app = Dash(external_stylesheets=[dbc.themes.SIMPLEX],suppress_callback_exceptions=True)

# dcc.storage default is storage_type = 'memory' this means the data is store until the page is refreshed

app.layout = html.Div([dcc.Location(id='page_location'),
                       dcc.Store(id='singleStored'),
                       dcc.Store(id='batchstore'), 
                       dcc.Store(id='ss2'),
                       dcc.Store(id='bs2'),
# nav bar makes the overall title at the top
                       dbc.NavbarSimple([ 
# navlink is the link to each of the tabs on the navbar
                    dbc.NavLink('Single Search', href='/',style = {'color': 'white'}), 
                    dbc.NavLink('Batch Count', href='/batch',style = {'color': 'white'}),
                    dbc.NavLink('Frequency', href='/frequency',style = {'color': 'white'}),
                    dbc.NavLink('Download', href='/download',style = {'color': 'white'}),
                    dbc.NavLink('Meta-analysis', href='/Meta-analysis',style = {'color': 'white'}),
                    dbc.NavLink('Raw Meta-analysis Data', href='/Raw',style = {'color': 'white'}),
                    dbc.NavLink('MetaData', href='/MetaData',style = {'color': 'white'})],      
                    brand='Aging Metabolite Database - AMDB', color = '#1A3E68', links_left = 'True', 
                    brand_style = {'color': '#FFCD00'},style={'width':'180%','margin-left': '-74.5%'}),
                       
                    dbc.Container(id='container_of_page')], style = {'margin-left': '-5%','width':'300%'})


# this is where the paths are selected. If single search is clicked it ouputs what the divs include

@app.callback(Output('container_of_page', 'children'), [Input('page_location', 'pathname')])
def pages(pathname):
    
# / is the path name they much has a backslash
    
    if pathname == '/':
        
        return html.Div(children = [html.H3('Search for information about a single value',
                            style = {'textAlign':'center','margin-left': '-250%','padding': '1.1%'}), 
                        html.Div([dcc.Dropdown(id='singledropdown',
                            options=[{'label': i, 'value': i} for i in MainData],value='Organ')],
                            style={'width': '17%','margin-left': '-120%'}),
 # the rows dont control what is typed but rather what you can see
                        dcc.Textarea(id = 'singleTextarea', rows = '1',
                            placeholder='Enter a single value here (Example: blood)', spellCheck='true',
                            style={'width': '50%','textAlign':'left', 'color': 
                       '#696969','margin-left': '-100%', 'padding': '1.1%','margin-top': '-3%'}), html.Br(), html.Br(),
                        html.Button(dcc.Link('Submit', href='/page-2', 
                            style = {'color':'black','text-decoration': 'none'}),style = {'margin-left': '-100%'}),
                        html.H3(instruction,style={'width': '115%','font-size': '19px','margin-left': '-127%'})
])
                            

    if pathname == '/page-2':
        
        return html.Div(html.Div(id='textareaout', 
                                 style={'whiteSpace': 'pre-line','margin-left': '-126%','padding': '1.1%'}))

    
    if pathname == '/batch':
        
        return html.Div([html.H3('Batch search for the frequency of multiple values',
                    style = {'textAlign':'center','margin-left': '-243%','padding': '1.1%'}), 
                 html.Div([dcc.Dropdown(id='batchdropdown',
                        options=[{'label': i, 'value': i} for i in MainData],value='Organ')],
                         style={'width': '17%','margin-left': '-120%'}),        
                dcc.Textarea(id = 'batchsearch',
                    placeholder= (batch_placeholder), spellCheck='true',
                    style={'width': '60%', 'height': '300px','textAlign':'left', 'color': 
                       '#696969','padding': '1.1%','margin-left': '-100%','margin-top': '-3%'}),
                    html.Br(), html.Br(),
                    html.Button(dcc.Link('Submit', href='/batch2',
# text decoration none is because there is an underline by default for links
                    style = {'color':'black','text-decoration': 'none'}),style = {'margin-left': '-100%'})
])
       

    if pathname == '/batch2':
        
        return html.Div(html.Div(id='batchout', style={'whiteSpace': 'pre-line','margin-left': '-112%','padding': '1.1%',
                                                      'width':'80%'}))
                         
    if pathname == '/frequency':
#padding puts space right after the div so it isnt so close        
        return html.Div([html.H3(
            'The following chart shows how often each value appears',style={'padding': '1.1%','margin-left': '-98%'}),
                         html.Div([dcc.Dropdown(id='demo-dropdown',
                            options=[{'label': i, 'value': i} for i in MainData],multi=True,value='Organ',
                            style={'width': '100%','background-color': '#F2F2F2'})], 
                            style={'width': '19%','margin-left': '-123%'}),
                       html.Div(id='dd-output-container',
                            style={'margin-left': '-99%','margin-top': '-3%','width':'70%','font-size': '18px'})
])
    
    if pathname == "/download":
        
        return html.Div([html.H3('Download the database',
                        style={'padding': '1.1%','margin-left': '-85%'}),
# I have only seen records/name/id used so I think its manditory
                         html.Div([dash_table.DataTable(MainData.to_dict('records'),[{'name': i, 'id': i} for i in MainData],
                            export_format="csv",
                            style_cell={'textAlign': 'left','font-family':'Open Sans','background-color': '#F2F2F2'},
                            style_data={'maxWidth':'275px'},
                            filter_action='native',
                            filter_options={'case':'insensitive','placeholder_text':'Type here to filter'},
                            sort_action='native',
                            sort_mode='multi')],
                            style={'margin-left': '-125%','margin-top': '-2.5%'})
])

    if pathname == "/Meta-analysis":
        
        return html.Div([html.H3('This combines multiple studies together to create a meta-analysis',
                        style={'padding': '1.1%','margin-left': '-85%'}),                         
                        html.Div([ html.Div([dcc.Dropdown(MetaAnalysisData.comparison.unique(), id='MetaDropdown', 
                            value=MetaAnalysisData.comparison.unique()[0],
                            style={'width': '100%','background-color': '#F2F2F2'})], 
                    style={'width': '35%'}),

                        html.Div(id = 'metatable', style={'margin-top': '1%'}),
                        html.Div(dcc.Graph(id='metafigure'),style={'margin-left': '23%','margin-top': '3%'})

])])

    
    if pathname == "/MetaData":
        
        return html.Div([html.H3('This tab shows all meta data that was collected for this database.',
                        style={'padding': '1.1%','margin-left': '-85%'}),
# I have only seen records/name/id used so I think its manditory
                         html.Div([dash_table.DataTable(MetaData.to_dict('records'),[{'name': i, 'id': i} for i in MetaData],
                            export_format="csv",
                            style_cell={'textAlign': 'left','font-family':'Open Sans','background-color': '#F2F2F2'},
                            style_data={'maxWidth':'275px'},
                            filter_action='native',
                            filter_options={'case':'insensitive','placeholder_text':'Type here to filter'},
                            sort_action='native',
                            sort_mode='multi')],
                            style={'margin-left': '-125%','margin-top': '-2.5%'})
])
    
    
    if pathname == "/Raw":
        
        return html.Div([html.H3('To create the meta-analysis multiple data point were combined. ',
                'This tab is used to show how these datapoints were combined.',
                        style={'padding': '1.1%','margin-left': '-85%'}),
# I have only seen records/name/id used so I think its manditory
                         html.Div([dash_table.DataTable(CombineData.to_dict('records'),[{'name': i, 'id': i} for i in CombineData],
                            export_format="csv",
                            style_cell={'textAlign': 'left','font-family':'Open Sans','background-color': '#F2F2F2'},
                            style_data={'maxWidth':'275px'},
                            filter_action='native',
                            filter_options={'case':'insensitive','placeholder_text':'Type here to filter'},
                            sort_action='native',
                            sort_mode='multi')],
                            style={'margin-left': '-125%','margin-top': '-2.5%'})
])
    

                           
@app.callback(
    Output('dd-output-container', 'children'),
    Input('demo-dropdown', 'value')
)
def update_output(value):
    frequency = MainData[value].value_counts()
# frequency takes in the value (selected column) and does the group by
    data = {'Count': frequency}
# data is the output that will be put into the dataframe
    indexReset = pd.DataFrame(data).reset_index()
# the value column will dissapear as it is the index column so it has to be reset 
# reset is when the new index is 0,1,2 instead of citrate, protein, etc
    finalOutput = indexReset.rename(columns={'index': value})
# dash only allows dictionaries to be the datatable so it has to be converted 
    return dash_table.DataTable(finalOutput.to_dict('records'),[{'name': i, 'id': i} for i in finalOutput],
                            export_format="csv",
                            style_cell={'textAlign': 'left','font-family':'Open Sans','background-color': '#F2F2F2'},
                            filter_action='native',
                            filter_options={'case':'insensitive','placeholder_text':'Type here to filter'},
                            page_size=20,
                            sort_action='native',
                            sort_mode='multi')
            

@app.callback(
        Output('singleStored','data'),
        [Input('singleTextarea','value')])

@app.callback(
        Output('batchstore','data'),
        [Input('batchsearch','value')])

@app.callback(
        Output('ss2','data'),
        [Input('singledropdown', 'value')])

@app.callback(
        Output('bs2','data'),
        [Input('batchdropdown', 'value')])
        
# this function just takes the typed words from the textArea and stores it into memory
def return_name(value):
    return value


@app.callback(
    Output('textareaout', 'children'),
    [Input('singleStored', 'data'),Input('ss2','data')])



def singleSearch(data,value):
    
    dataframe = Search(data,value)   
             
    return dash_table.DataTable(dataframe.to_dict('records'),[{'name': i, 'id': i} for i in dataframe],export_format="csv",
                               style_cell={'textAlign': 'left','font-family':'Open Sans','background-color': '#F2F2F2'},
                                filter_action='native',
                                filter_options={'case':'insensitive','placeholder_text':'Type here to filter'},
                                sort_action='native',
                                sort_mode='multi'
)
                     


@app.callback(
    Output('batchout', 'children'),
    [Input('batchstore', 'data'),Input('bs2','data')])

# this function takes the text area that was stored on the new page, turns it into a list
# and splits it by row so each row is in its own string
# ['a','b','c']
# this is possible beacause textarea puts an extra space at the end of a row so the split '__'
# this also allows for a space in the name

def batchSearch(data,value):
    
    dataframe = Search(data,value) 

    frequency = dataframe[value].value_counts()
#    frequency = dataframe.value_counts()
    newdata = {'Count': frequency}
    indexReset = pd.DataFrame(newdata).reset_index()
             
    return dash_table.DataTable(indexReset.to_dict('records'),[{'name': i, 'id': i} for i in indexReset],
                            export_format="csv",
                            style_cell={'textAlign': 'left','font-family':'Open Sans','background-color': '#F2F2F2'},
                            filter_action='native',
                            filter_options={'case':'insensitive','placeholder_text':'Type here to filter'},
                            sort_action='native',
                            sort_mode='multi') 


#meta analysis call backs                                
@app.callback(
    Output('metafigure', 'figure'),
    [Input('MetaDropdown', 'value')]
)
def MetaAnlysisFigure(selected_metabolite):
    # Call the PlotlyFigure function with the selected metabolite
    thedata = Data(selected_metabolite)
    return PlotlyFigure(thedata,selected_metabolite)[1]




@app.callback(
    Output('metatable', 'children'),
    [Input('MetaDropdown', 'value')]
)
def MetaAnalysisTable(selected_metabolite):
    # Call the PlotlyFigure function with the selected metabolite
    thedata = Data(selected_metabolite)
    meta =  PlotlyFigure(thedata,selected_metabolite)[0]
    return dash_table.DataTable(meta.to_dict('records'),[{'name': i, 'id': i} for i in meta],
                export_format="csv",
                style_cell={'textAlign': 'left','font-family':'Open Sans','background-color': '#F2F2F2'})

                             
                                 

# use_reloader = False is needed when ran on jupyter or debug wont work
# jupyter_mode="external" creates a pop out instead of running internally on jupyter

                                              
if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False, jupyter_mode="external") 


# In[ ]:




