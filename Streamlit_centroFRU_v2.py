import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import re
from matplotlib.patches import ConnectionPatch
import plotly.express as px
from st_keyup import st_keyup


# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False
    st.session_state.horizontal = False


#data = pd.read_excel("C:/Users/Mandich/Desktop/Abstract_Cla/FRU_CP_fin.xlsx", sheet_name="tot")

st.title("CentroFRU")
#@st.cache
st.write("Use your own XLSX!")
oas_file = st.file_uploader(label="Upload an XLSX",type=["xlsx"],accept_multiple_files=False)
#def decode_uploaded_file(oas_file: UploadedFile) -> str:
#    return oas_file.read().decode()
#raw_oas = decode_uploaded_file(oas_file) --> to show raw data (code)
if oas_file is not None:
        #@st.cache_data
        data=pd.read_excel(oas_file, sheet_name="tot")
        data = data[data.columns[:-6]]
        data.rename(columns={'Unnamed: 3':'Names','altri test genetici': 'Others', 'Unnamed: 25':'Notes',
                             'Risultato':'Karyo_res','Risultato.1':'microdel_res','Risultato.2':'FRAXA_res','Risultato.3':'Others_res'}, inplace=True)
        data['Cariotipo_num'] = pd.Series(np.where(data.Cariotipo.values == 'si', 1, 0), data.index)
        data['microdel_Y_num'] = pd.Series(np.where(data['microdel Y'].values == 'si', 1, 0), data.index)
        data['FRAXA_num'] = pd.Series(np.where(data.FRAXA.values == 'si', 1, 0), data.index)
        data['Others_num'] = pd.Series(np.where(data['Others'].values == 'si', 1, 0), data.index)
        data['tot_prescribed_exams_num'] = data['Cariotipo_num'] + data['FRAXA_num'] + data['Others_num'] + data['microdel_Y_num']
        data['Karyo_res_num'] = pd.Series(np.where(data.Karyo_res.values == 'alt', 1, 0), data.index)
        data['microdel_res_num'] = pd.Series(np.where(data['microdel_res'].values == 'alt', 1, 0), data.index)
        data['FRAXA_res_num'] = pd.Series(np.where(data.FRAXA_res.values == 'alt', 1, 0), data.index)
        data['Others_res_num'] = pd.Series(np.where(data['Others_res'].values == 'alt', 1, 0), data.index)
        data['tot_alterate_exams_num'] = data['Karyo_res_num'] + data['microdel_res_num'] + data['FRAXA_res_num'] + data['Others_res_num']
        data['spermiogramma'].replace(np.nan, '', inplace=True)
        data['esami ormonali'].replace(np.nan, '', inplace=True)
        data['esami_alterati'] = data['spermiogramma'] + data['esami ormonali']
        data.INF.replace('', 'A', inplace=True)
        data.FAM.replace('', 'A', inplace=True)
        data['INF'].replace(np.nan, "A", inplace=True)
        data['FAM'].replace(np.nan, "A", inplace=True)
        data.INF = [e.strip(' ') for e in data.INF]
        data.FAM = [e.strip(' ') for e in data.FAM]
        data['FamHis_Exams'] = np.where((data.INF == 'x') & (data.FAM == 'x'), 'INF_FAM', np.where((data.INF == 'x') & (data.FAM == 'A'), 'INF_noFAM',
                                         np.where((data.INF == 'A') & (data.FAM == 'x'), 'FAM_noINF', 'NA')))
        data['INF'].replace('A', np.nan, inplace=True)
        data['FAM'].replace('A', np.nan, inplace=True)
        data['alt_res'] = pd.Series(np.where(data['tot_alterate_exams_num'].values == 0, 'negative_res', 'positive_res'))
        ################################
        on = st.toggle('Show input dataframe')
        if on:
                st.dataframe(data)

        tab1, tab2 = st.tabs(["Exploring", "Filtering"])

        with tab1:
                st.title("Explore db")
                graph_option = st.selectbox('What would you like to inspect graphically?',
                                            ("Numero esami alterati per sesso", "Quantità esami prescritti", "Quantità esami alterati", "Spettro esami alterati",
                                             "Percentage of DI/DS/MALF per family history", "Number of alt. exams per couple",
                                             "N of karyo and FRAXA divided by sex and by sperm/esami ormonali", "Esami patologici"))

                ###################################################### 1 N of alterate exam divided by sex ###################################################

                if graph_option == "Numero esami alterati per sesso":
                        sperm_ok = data.groupby('spermiogramma').count()['num']['ok']
                        s_dict = {'tot': len(data) / 2, 'alt': len(data) / 2 - sperm_ok, 'ok': sperm_ok}
                        sperm = pd.Series(s_dict)

                        orm_ok = data.groupby('esami ormonali').count()['num']['ok']
                        o_dict = {'tot': len(data) / 2, 'alt': len(data) / 2 - orm_ok, 'ok': orm_ok}
                        orm = pd.Series(o_dict)

                        tot = pd.DataFrame([sperm, orm], index=['M', 'F'])
                        # graph

                        st.subheader(f"Number of alterate exams by sex N={len(data)}")
                        st.bar_chart(tot, y=['alt', 'ok'])  # , color=['#008b8b','#b22222']

                        del orm, sperm, sperm_ok, orm_ok, s_dict, o_dict, tot

                ##############################################################################################################################################

                ############# 2 Q.ty prescribed exams (Caryo, FRAXA, Other) divided by fam history or No fam history but alterated exams #####################

                if graph_option == "Quantità esami prescritti":
                        tot = data[data.tot_prescribed_exams_num != 0].groupby("FamHis_Exams")["tot_prescribed_exams_num"].sum().sum()
                        fig, ax = plt.subplots(figsize=(12, 10))
                        ax.pie(data[data.tot_prescribed_exams_num != 0].groupby("FamHis_Exams")["tot_prescribed_exams_num"].sum(),
                               autopct=lambda p: "{:.1f}%\n(N={:.0f})".format(p, p / 100 * tot),
                               labels=data.groupby("FamHis_Exams")["tot_prescribed_exams_num"].sum().index,
                               radius=1, labeldistance=1.2,
                               textprops={'size': 'medium', 'weight': 'bold'}, wedgeprops={'edgecolor': 'white'})
                        ax.set_title(f"Number of prescribed exams according to criteria\nN_tot_prescribed={tot}", fontsize=15)
                        st.pyplot(fig) #colors=['darkcyan', 'darkred'],

                        up = st.toggle('Show input dataframe', key="2")
                        if up:
                                st.dataframe(data[data.tot_prescribed_exams_num != 0].reset_index().drop(columns=['index']))

                ##############################################################################################################################################

                ############################################## 3 Q.ty "alt" exams / tot (prescritti) #########################################################

                if graph_option == "Quantità esami alterati":
                        for_Res = data[data.COMPLETO != "no"]
                        for_Res = for_Res.groupby('alt_res')[['tot_alterate_exams_num', "tot_prescribed_exams_num"]].sum()
                        for_Res.loc['negative_res', 'tot_alterate_exams_num'] = for_Res['tot_prescribed_exams_num'].sum() - for_Res.loc['positive_res', 'tot_alterate_exams_num']
                        for_Res.drop(columns='tot_prescribed_exams_num', inplace=True)
                        # first edit df to get col q/ unique exams
                        totale_prescritti_pos = data[data.alt_res == "positive_res"]['tot_prescribed_exams_num'].sum()
                        data['Cariotipo'] = pd.Series(np.where(data.Cariotipo.values == 'si', "Karyotype", ''), data.index)
                        data['microdel Y'] = pd.Series(np.where(data['microdel Y'].values == 'si', "microdelY", ''), data.index)
                        data['FRAXA'] = pd.Series(np.where(data.FRAXA.values == 'si', "FRAXA", ''), data.index)
                        data['Others'] = pd.Series(np.where(data.Others.values == 'si', "Others", ''), data.index)
                        data['prescribed_exams'] = data['Cariotipo'].astype(str) + "-" + data['microdel Y'].astype(str) + "-" + data['FRAXA'].astype(str)+ "-" + data['Others'].astype(str)
                        exams=data[data.tot_prescribed_exams_num !=0]
                        # exams['prescribed_exams'].unique()
                        exams.prescribed_exams = [re.sub('-+', '-', e) for e in exams.prescribed_exams]
                        exams.prescribed_exams = [re.sub(r'^-', '', e) for e in exams.prescribed_exams]
                        exams.prescribed_exams = [re.sub(r'-$', '', e) for e in exams.prescribed_exams]
                        exams.prescribed_exams.replace('', np.nan, inplace=True)
                        #for_POS = exams[exams.alt_res=="positive_res"].groupby('prescribed_exams').sum()[['tot_prescribed_exams_num','tot_alterate_exams_num']]
                        for_POS = exams[exams.alt_res == "positive_res"]
                        new_POS = pd.DataFrame(index=['Cariotipo', 'microdel_Y', 'FRAXA', 'Others'], columns=['N_exams'])
                        new_POS.at['Cariotipo', 'N_exams'] = for_POS['Karyo_res_num'].sum()
                        new_POS.at['microdel_Y', 'N_exams'] = for_POS['microdel_res_num'].sum()
                        new_POS.at['FRAXA', 'N_exams'] = for_POS['FRAXA_res_num'].sum()
                        new_POS.at['Others', 'N_exams'] = for_POS['Others_res_num'].sum()

                        fig = plt.figure(figsize=(10.79, 6.075))
                        ax1 = fig.add_subplot(121)
                        ax2 = fig.add_subplot(122)
                        fig.subplots_adjust(wspace=0)
                        explode = [0.05, 0]
                        # rotate so that first wedge is split by the x-axis
                        angle = 180 * 0.0525  #0.0825
                        ax1.pie(for_Res["tot_alterate_exams_num"], ######## USE THE COLUMN WITH MORE VALUES IN THIS CASE IS SEX
                                autopct=lambda p: "{:.1f}%\n(N={:.0f})".format(p, p / 100 * for_Res["tot_alterate_exams_num"].sum()),
                                startangle=angle,
                                labels=for_Res["tot_alterate_exams_num"].index, explode=explode,
                                textprops={'size': 'medium', 'weight': 'bold'},
                                labeldistance=1.05,colors=['darkcyan', 'darkred'])
                        width = .2
                        ax2.pie(new_POS['N_exams'],
                                autopct=lambda p: "{:.1f}%\n(N={:.0f})".format(p, p / 100 * new_POS['N_exams'].sum()) if p > 4 else '',
                                startangle=angle, labels=new_POS.index,
                                radius=0.75,
                                textprops={'size': 'smaller', 'weight': 'bold'},
                                labeldistance=1.1)
                        n_sub = exams.groupby("alt_res").count()["sex"]['positive_res']
                        ax1.set_title(
                                f'Number of positive results in prescribed exams\nN_tot_subject={len(exams)}\nN_tot_exams={exams.tot_prescribed_exams_num.sum()}')
                        ax2.set_title(f"Type of exams per positive results\nN_tot_subject={n_sub}\nN_tot_exams={totale_prescritti_pos}")
                        # get the wedge data
                        theta1, theta2 = ax1.patches[0].theta1, ax1.patches[0].theta2
                        center, r = ax1.patches[0].center, ax1.patches[0].r
                        # draw top connecting line
                        x = r * np.cos(np.pi / 180 * theta2) + center[0]
                        y = np.sin(np.pi / 180 * theta2) + center[1]
                        con = ConnectionPatch(xyA=(- width / 2, -.8), xyB=(x, y),
                                              coordsA="data", coordsB="data", axesA=ax2, axesB=ax1)
                        con.set_color([0, 0, 0])
                        con.set_linewidth(1)
                        ax2.add_artist(con)
                        # draw bottom connecting line
                        x = r * np.cos(np.pi / 180 * theta1) + center[0]
                        y = np.sin(np.pi / 180 * theta1) + center[1]
                        con = ConnectionPatch(xyA=(- width / 2, .8), xyB=(x, y), coordsA="data",
                                              coordsB="data", axesA=ax2, axesB=ax1)
                        con.set_color([0, 0, 0])
                        ax2.add_artist(con)
                        con.set_linewidth(1)
                        st.pyplot(fig)

                        up = st.toggle('Show input dataframe', key="3")
                        if up:
                                st.dataframe(exams.reset_index().drop(columns=['index']))

                        del angle, ax1, ax2, center, con, explode, fig, r, theta1, theta2, width, x, y, n_sub, totale_prescritti_pos

                ##############################################################################################################################################

                #################################### 4 Percentage of DI/DS/MALF etc. by FAM history but examination okay #####################################

                if graph_option == "Spettro esami alterati":
                        left_column, right_column = st.columns(2)
                        with left_column:
                                chosen = st.radio('Select group of interest:', ('Male', 'Female'))
                        with right_column:
                                if chosen == "Male":
                                        male = data[data['sex']=="M"]
                                        fig, ax = plt.subplots(figsize=(12, 10))
                                        ax.pie(male[male.esami_alterati != ''].groupby("esami_alterati").count()["sex"],
                                               autopct=lambda p: "{:.1f}%\n(N={:.0f})".format(p, p / 100 * len(male[male.esami_alterati != '']))if p>2 else '',
                                               labels=male[male.esami_alterati != ''].groupby("esami_alterati").count()["sex"].index,
                                               radius=1, labeldistance=1.2,
                                               textprops={'size': 'medium', 'weight': 'bold'}, wedgeprops={'edgecolor': 'white'})
                                        ax.set_title(f"Distribution of altered exams in {chosen} only\nN_tot_subject={len(male[male.esami_alterati != ''])}", fontsize=15)
                                        st.pyplot(fig)
                                        up = st.toggle('Show input dataframe', key="4_M")
                                        if up:
                                                st.dataframe(male[male.esami_alterati != ''].reset_index().drop(columns="index"))
                                elif chosen == "Female":
                                        female = data[data['sex'] == "F"]
                                        fig, ax = plt.subplots(figsize=(12, 10))
                                        ax.pie(female[female.esami_alterati != ''].groupby("esami_alterati").count()["sex"],
                                               autopct=lambda p: "{:.1f}%\n(N={:.0f})".format(p, p / 100 * len(
                                                       female[female.esami_alterati != ''])) if p > 2 else '',
                                               labels=female[female.esami_alterati != ''].groupby("esami_alterati").count()["sex"].index,
                                               radius=1, labeldistance=1.2,
                                               textprops={'size': 'medium', 'weight': 'bold'}, wedgeprops={'edgecolor': 'white'})
                                        ax.set_title(
                                                f"Distribution of altered exams in {chosen} only\nN_tot_subject={len(female[female.esami_alterati != ''])}",
                                                fontsize=15)
                                        st.pyplot(fig)
                                        up = st.toggle('Show input dataframe', key="4_F")
                                        if up:
                                                st.dataframe(female[female.esami_alterati != ''].reset_index().drop(columns="index"))

                ##############################################################################################################################################

                #################################### 4 Percentage of DI/DS/MALF etc. by FAM history but examination okay #####################################

                if graph_option == "Percentage of DI/DS/MALF per family history":

                        famHis = data[data['FamHis_Exams'] == "FAM_noINF"]
                        col2mod = ['DI/DS/AUT', 'MALF', 'MP', 'PA']
                        for c in col2mod:
                                famHis[c].fillna('', inplace=True)

                        famHis['FamHistory'] = famHis['DI/DS/AUT'].astype(str) + "+" + famHis['MALF'].astype(str) + "+" + famHis['MP'].astype(
                                str) + "+" + famHis['PA'].astype(str)

                        famHis.FamHistory = [e.lstrip('^+') for e in famHis.FamHistory]
                        famHis.FamHistory = [e.rstrip(r'+$') for e in famHis.FamHistory]
                        famHis.FamHistory = [re.sub(r'\++', '+', e) for e in famHis.FamHistory]
                        famHis.FamHistory.replace('', 'Altro', inplace=True)

                        fig, ax = plt.subplots(figsize=(12, 10))
                        wedges, texts, autotexts = ax.pie(famHis.groupby('FamHistory').count().reset_index()['sex'],
                                                          autopct=lambda p: "{:.1f}%\n(N={:.0f})".format(p, p / 100 * famHis.groupby(
                                                                  'FamHistory').count().reset_index()['sex'].sum()),
                                                          labels=famHis.groupby('FamHistory').count().reset_index()['FamHistory'],
                                                          radius=1, labeldistance=1.2,
                                                          textprops={'size': 'medium', 'weight': 'bold'}, wedgeprops={'edgecolor': 'white'})
                        ax.set_title(
                                f"Alteration distribution according to family history and normal exams\
                                        \nN_tot={famHis.groupby('FamHistory').count().reset_index()['sex'].sum()}",
                                fontsize=15)
                        ## the following removes labels and percentage below 2%
                        threshold = 2
                        for label, pct_label in zip(texts, autotexts):
                                pct_value = pct_label.get_text().rstrip(r"%\n(N=1234567890.)")
                                pct_value = pct_value.rstrip("%\n")
                                if float(pct_value) < threshold:
                                        label.set_text('')
                                        pct_label.set_text('')
                        st.pyplot(fig)

                        up = st.toggle('Show input dataframe', key="5")
                        if up:
                                st.dataframe(famHis.reset_index().drop(columns="index"))

                        del threshold, pct_label, wedges, autotexts, texts, pct_value, label, famHis

                ########################################### 5 number of changes (i.e. "alt" in exams) per couple #############################################

                if graph_option == "Number of alt. exams per couple":
                        import math

                        couple = data.copy()
                        couple['alt_res'] = pd.Series(np.where(couple['tot_alterate_exams_num'].values == 0, 'negative_res', 'positive_res'))
                        couple = couple[couple.COMPLETO != "no"]
                        # couple['alter'] = couple[['spermiogramma', 'esami ormonali']].isin(['alt']).any(axis=1)
                        # coup_numb = (couple[couple['alter'] == True]).reset_index()
                        coup_numb = couple[couple['alt_res'] != "negative_res"].reset_index()
                        get_index_coup = list(coup_numb['num'])
                        couple_df = couple[couple['num'].isin(get_index_coup)]
                        no_changed_coup = couple[~couple['num'].isin(get_index_coup)]

                        couple_df['path'] = 'Path_changed'
                        no_changed_coup['path'] = 'No_change_in_Path'

                        tot_couple = pd.concat([couple_df, no_changed_coup], axis=0)

                        fig, ax = plt.subplots(figsize=(10, 8))
                        ax.pie(tot_couple.groupby('path').count()["num"] / 2,
                               autopct=lambda p: "{:.1f}%\n(N={:.0f})".format(p, p / 100 * len(tot_couple) / 2),
                               labels=tot_couple.groupby('path').count()["num"].index,
                               colors=['darkcyan', 'darkred'], radius=1, labeldistance=1.2,
                               textprops={'size': 'medium', 'weight': 'bold'}, wedgeprops={'edgecolor': 'white'})
                        ax.set_title(f"Number of results per couple\nN_tot={math.floor(len(tot_couple) / 2)}", fontsize=15)
                        st.pyplot(fig)

                        up = st.toggle('Show input dataframe', key="5")
                        if up:
                                st.dataframe(tot_couple.reset_index().drop(columns="index"))

                        del tot_couple, couple, couple_df, coup_numb, get_index_coup, no_changed_coup

                ##############################################################################################################################################

                #################################### 6 N of karyo and FRAXA divided by sex and by sperm/esami ormonali #######################################

                if graph_option == "N of karyo and FRAXA divided by sex and by sperm/esami ormonali":
                        left_column, right_column = st.columns(2)
                        with left_column:
                                chosen = st.radio('Select variable of interest',
                                                  ("All", "Male", "Female", "FRAXA"))
                                # st.write(f"You are in {chosen} house!")
                        with right_column:
                                ### probably needed two nested pie plots - try in chunks
                                # first edit df to get col q/ unique exams
                                data['Cariotipo'] = pd.Series(np.where(data.Cariotipo.values == 'si', "Karyotype", ''), data.index)
                                data['microdel Y'] = pd.Series(np.where(data['microdel Y'].values == 'si', "microdel_Y", ''), data.index)
                                data['FRAXA'] = pd.Series(np.where(data.FRAXA.values == 'si', "FRAXA", ''), data.index)
                                data['Others'] = pd.Series(np.where(data.Others.values == 'si', "Others", ''), data.index)
                                data['prescribed_exams'] = data['Cariotipo'].astype(str) + "-" + data['microdel Y'].astype(str) + "-" + data['FRAXA'].astype(str) + "-" + data[
                                        'Others'].astype(str)
                                # exams['prescribed_exams'].unique()
                                data.prescribed_exams = [re.sub('-+', '-', e) for e in data.prescribed_exams]
                                data.prescribed_exams = [re.sub(r'^-', '', e) for e in data.prescribed_exams]
                                data.prescribed_exams = [re.sub(r'-$', '', e) for e in data.prescribed_exams]
                                data.prescribed_exams.replace('', np.nan, inplace=True)

                                exams = data[data['COMPLETO'] != "no"]
                                new_ex = exams.copy()
                                new_ex_M = new_ex[new_ex['sex'] == "M"]
                                new_ex_F = new_ex[new_ex['sex'] == "F"]

                                new_ex_M = new_ex_M[new_ex_M['prescribed_exams'] == "Karyotype"]

                                new_ex_karyo = new_ex[new_ex['Cariotipo'] == "Karyotype"]
                                new_ex_karyo = new_ex_karyo[new_ex_karyo['FamHis_Exams']!="NA"]
                                karyo_sunburst = new_ex_karyo.groupby(['FamHis_Exams', 'sex']).count().reset_index()

                                if chosen == "All":
                                        fig = px.sunburst(karyo_sunburst, path=["FamHis_Exams", "sex"], values="num", color="FamHis_Exams",
                                                          names="sex",
                                                          color_discrete_map={'FAM_noAlt_exams': 'darkred', 'noFAM_alt_exams': 'darkcyan'},
                                                          width=500,
                                                          height=500)
                                        fig.update_traces(textinfo="label+value+percent parent")
                                        fig.update_traces(texttemplate='<b>%{label}</b><br><b>N=%{value}</b><br><b>%{percentParent}</b>')
                                        fig.update_traces(hovertemplate='<b>%{label}</b><br>N_tot=%{value}<br>', sort=False, rotation=90)
                                        fig.update_layout(uniformtext=dict(minsize=5))  # , mode='hide'
                                        fig.update_layout(title_font_color="black", title_font_size=20)
                                        fig.update_traces(insidetextorientation='horizontal')
                                        fig.update_layout(
                                                title_text=f'Karyotype examinations<br>\nN_prescribed_karyotype_tot={karyo_sunburst.num.sum()}')
                                        st.plotly_chart(fig)
                                elif chosen == "Male":
                                        fig, ax = plt.subplots(figsize=(12, 10))
                                        karyo_sunburst = karyo_sunburst[karyo_sunburst.FamHis_Exams != "NA"]
                                        ax.pie(karyo_sunburst[karyo_sunburst["sex"] == "M"]['Names'],
                                               autopct=lambda p: "{:.1f}%\n(N={:.0f})".format(p, p / 100 *
                                                                                              karyo_sunburst[(karyo_sunburst['sex'] == 'M') &
                                                                                                             (karyo_sunburst.FamHis_Exams != 'NA')][
                                                                                                      'Names'].sum()),
                                               labels=karyo_sunburst[karyo_sunburst["sex"] == "M"]["FamHis_Exams"],
                                               radius=1, labeldistance=1.2,
                                               textprops={'size': 'medium', 'weight': 'bold'}, wedgeprops={'edgecolor': 'white'})
                                        ax.set_title(
                                                f"Number of prescribed karyoptype in Male only\nN_tot="
                                                f"{karyo_sunburst[(karyo_sunburst['sex'] == 'M') & (karyo_sunburst.FamHis_Exams != 'NA')]['Names'].sum()}",
                                                fontsize=15)
                                        st.pyplot(fig)
                                elif chosen == "Female":
                                        fig, ax = plt.subplots(figsize=(12, 10))
                                        karyo_sunburst = karyo_sunburst[karyo_sunburst.FamHis_Exams != "NA"]
                                        ax.pie(karyo_sunburst[karyo_sunburst["sex"] == "F"]['Names'],
                                               autopct=lambda p: "{:.1f}%\n(N={:.0f})".format(p, p / 100 *
                                                                                              karyo_sunburst[karyo_sunburst['sex'] == 'F'][
                                                                                                      'Names'].sum()),
                                               labels=karyo_sunburst[karyo_sunburst["sex"] == "F"]["FamHis_Exams"],
                                               radius=1, labeldistance=1.2,
                                               textprops={'size': 'medium', 'weight': 'bold'}, wedgeprops={'edgecolor': 'white'})
                                        ax.set_title(
                                                f"Number of prescribed karyoptype in Female only\nN_tot="
                                                f"{karyo_sunburst[(karyo_sunburst['sex'] == 'F') & (karyo_sunburst.FamHis_Exams != 'NA')]['Names'].sum()}",
                                                fontsize=15)
                                        st.pyplot(fig)
                                else:
                                        fraxa = new_ex[new_ex.FRAXA == "FRAXA"]

                                        fig, ax = plt.subplots(figsize=(12, 10))
                                        ax.pie(fraxa[fraxa.FamHis_Exams != "NA"].groupby("FamHis_Exams").count()['sex'],
                                               autopct=lambda p: "{:.1f}%\n(N={:.0f})".format(p,
                                                                                              p / 100 *
                                                                                              fraxa[fraxa.FamHis_Exams != "NA"].groupby("FamHis_Exams").count()['sex'].sum()),
                                               labels=fraxa[fraxa.FamHis_Exams != "NA"].groupby("FamHis_Exams").count().index,
                                               radius=1, labeldistance=1.2,
                                               textprops={'size': 'medium', 'weight': 'bold'}, wedgeprops={'edgecolor': 'white'})
                                        ax.set_title(
                                                f"Number of prescribed FRAXA in Female only\nN_tot="
                                                f"{fraxa[fraxa.FamHis_Exams != 'NA'].groupby('FamHis_Exams').count()['sex'].sum()}",
                                                fontsize=15)
                                        st.pyplot(fig)

                ##############################################################################################################################################

                #################################### 6 N of karyo and FRAXA divided by sex and by sperm/esami ormonali #######################################

                if graph_option == "Esami patologici":
                        path = ['Familiarità'] * 7 + ['Infertilità'] * 14
                        fam_exam = ['CARIOTIPO'] * 3 + ['CFTR'] * 1 + ['FRAXA'] * 1 + ['GJB2'] * 2
                        inf_exam = ['CARIOTIPO'] * 6 + ['FRAXA'] * 1 + ['CFTR'] * 1 + ['GLOBO'] * 1 + ['MICRODEL AZF'] * 5
                        exam = fam_exam + inf_exam

                        data_2 = pd.DataFrame({'path': path, 'exam': exam})
                        data_2['conte'] = 1

                        data_2 = data_2.groupby(['path', 'exam']).count().reset_index()

                        fig = px.sunburst(data_2, path=['path', 'exam'], values="conte", names="exam", color="path",
                                          color_discrete_map={'Familiarità': '#0179A3', 'Infertilità': '#940000'}, width=750, height=750)
                        fig.update_traces(textinfo="label+value+percent parent")
                        fig.update_traces(texttemplate='<b>%{label}</b><br><b>N=%{value}</b><br><b>%{percentParent}</b>')
                        fig.update_traces(hovertemplate='<b>%{label}</b><br>N_tot=%{value}<br>', sort=False, rotation=90)
                        fig.update_layout(uniformtext=dict(minsize=5))  # , mode='hide'
                        fig.update_layout(title_font_color="black", title_font_size=32)
                        fig.update_traces(insidetextorientation='horizontal')
                        fig.update_layout(title_text=f'<b>ESAMI PATOLOGICI</b>\nN_tot={len(data_2)}')  # , title_x=0.2
                        st.plotly_chart(fig)

        with tab2:
                #st.write("Need 2 implement")

                st.title("Filter db")
                filter_option = st.selectbox('Select variable for subset:',
                                            ("Name", "Spermiogramma", "Esami ormonali", "Risultati alterazioni", "Reason of enrollement",
                                             "Type of alteration", "Other tests - What?", "Notes"))

                if filter_option == "Name":
                        st.write("Please insert your:", filter_option)
                        selection = st_keyup("Enter input")
                        selection = selection.lower()
                        st.write(data[data["Names"].str.contains(selection)])
                elif filter_option == "Spermiogramma":
                        sperm_alter = data[data["spermiogramma"] != '']['spermiogramma'].unique()
                        sce_spe = st.multiselect(label="Select alteration:", options=sperm_alter)
                        st.write('**N.B. blank removed (i.e. no alteration)**')
                        st.write(data[data.spermiogramma.isin(sce_spe)])
                        st.download_button(label="Download data as CSV",
                                           data=data[data.spermiogramma.isin(sce_spe)].to_csv(index=False, sep="\t"),
                                           key="d1", file_name="spermiogramma_filtered.txt")#, mime="text/csv"
                        del sperm_alter, sce_spe
                elif filter_option == "Esami ormonali":
                        orm_alter = data[data["esami ormonali"] != '']['esami ormonali'].unique()
                        sce_orm = st.multiselect(label="Select alteration:", options=orm_alter)
                        st.write('**N.B. blank removed (i.e. no alteration)**')
                        st.write(data[data['esami ormonali'].isin(sce_orm)])
                        st.download_button(label="Download data as CSV",
                                           data=data[data['esami ormonali'].isin(sce_orm)].to_csv(index=False, sep="\t"),
                                           key="d1", file_name="esami_romonali_filtered.txt")
                        del orm_alter, sce_orm
                elif filter_option == "Risultati alterazioni":
                        st.write("Please insert your alteration")
                        mut = st_keyup("Enter input")
                        simplified_data = data.copy()
                        simplified_data.dropna(subset=['alt'], inplace=True)
                        simplified_data = simplified_data.reset_index().drop(columns='index')
                        st.write("**It is case-sensitive, so be sure on what you type**")
                        st.write("Non-altered entries are removed by default")
                        st.write(simplified_data[simplified_data.alt.str.contains(mut)])
                        st.download_button(label="Download data as CSV",
                                           data=simplified_data[simplified_data.alt.str.contains(mut)].to_csv(index=False, sep="\t"),
                                           key="d1", file_name="Risultati_alterazioni_filtered.txt")
                        del simplified_data, mut
                elif filter_option == "Reason of enrollement":
                        reas_y = data[data.FamHis_Exams != 'NA']['FamHis_Exams'].unique()
                        sce_reas = st.multiselect(label="Select reason of enrollement:", options=reas_y)
                        st.write('**N.B. blank removed (i.e. no alteration)**')
                        st.write(data[data['FamHis_Exams'].isin(sce_reas)])
                        st.download_button(label="Download data as CSV",
                                           data=data[data['FamHis_Exams'].isin(sce_reas)].to_csv(index=False, sep="\t"),
                                           key="d1", file_name="Reason_enrollement_filtered.txt")
                        del reas_y, sce_reas
                elif filter_option == "Type of alteration":
                        famHis = data[data['FamHis_Exams'] == "FAM_noINF"]
                        col2mod = ['DI/DS/AUT', 'MALF', 'MP', 'PA']
                        for c in col2mod:
                                famHis[c].fillna('', inplace=True)

                        famHis['FamHistory'] = famHis['DI/DS/AUT'].astype(str) + "+" + famHis['MALF'].astype(str) + "+" + famHis['MP'].astype(
                                str) + "+" + famHis['PA'].astype(str)

                        famHis.FamHistory = [e.lstrip('^+') for e in famHis.FamHistory]
                        famHis.FamHistory = [e.rstrip(r'+$') for e in famHis.FamHistory]
                        famHis.FamHistory = [re.sub(r'\++', '+', e) for e in famHis.FamHistory]
                        famHis.FamHistory.replace('', 'Altro', inplace=True)
                        typeofex = famHis[famHis.FamHistory != 'NA']['FamHistory'].unique()
                        sce_typeofex = st.multiselect(label="Select type of alteration in those with family history:", options=typeofex)
                        st.write('**N.B. blank removed (i.e. no family history)**')
                        st.write(famHis[famHis.FamHistory.isin(sce_typeofex)].reset_index().drop(columns='index'))
                        st.download_button(label="Download data as CSV",
                                           data=famHis[famHis.FamHistory.isin(sce_typeofex)].reset_index().drop(columns='index').to_csv(index=False, sep="\t"),
                                           key="d1", file_name="Type_of_alteration_filtered.txt")
                        del famHis, typeofex, sce_typeofex, col2mod, c
                elif filter_option == "Other tests - What?":
                        st.write('**N.B. blank removed (i.e. no other tests)**')
                        st.write(data[~data.quali.isna()])
                else:
                        st.write('**N.B. blank removed (i.e. no Notes)**')
                        st.write(data[~data.Notes.isna()])

