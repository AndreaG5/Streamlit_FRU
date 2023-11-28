import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import re
from matplotlib.patches import ConnectionPatch
import plotly.express as px


# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False
    st.session_state.horizontal = False


#data = pd.read_excel("C:/Users/Mandich/Desktop/Abstract_Cla/Centro FRU_CP_my_edit.xlsx", sheet_name="tot")

st.title("Prova_CentroFRU")
#@st.cache
st.write("Use your own XLSX!")
oas_file = st.file_uploader(label="Upload an XLSX",type=["xlsx"],accept_multiple_files=False)
#def decode_uploaded_file(oas_file: UploadedFile) -> str:
#    return oas_file.read().decode()
#raw_oas = decode_uploaded_file(oas_file) --> to show raw data (code)
if oas_file is not None:
        #@st.cache_data
        data=pd.read_excel(oas_file, sheet_name="tot")
        data.rename(columns={'FAM + con esami ok': 'FamHis_Exams', 'altri test genetici': 'Others'}, inplace=True)
        data['Cariotipo_num'] = pd.Series(np.where(data.Cariotipo.values == 'si', 1, 0), data.index)
        data['FRAXA_num'] = pd.Series(np.where(data.FRAXA.values == 'si', 1, 0), data.index)
        data['Others_num'] = pd.Series(np.where(data['Others'].values == 'si', 1, 0), data.index)
        data['FamHis_Exams'] = pd.Series(np.where(data['FamHis_Exams'].values == 'si', 'FAM_noAlt_exams', 'noFAM_alt_exams'), data.index)
        data['tot_prescribed_exams_num'] = data['Cariotipo_num'] + data['FRAXA_num'] + data['Others_num']
        data['spermiogramma'].replace(np.nan, '', inplace=True)
        data['esami ormonali'].replace(np.nan, '', inplace=True)
        data['esami_alterati'] = data['spermiogramma'] + data['esami ormonali']
        st.dataframe(data)

        st.title("Explore db")
        graph_option = st.selectbox('What would you like to graphically inspect?',
                                    ("Numero esami alterati per sesso", "Quantità esami prescritti", "Quantità esami alterati",
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

                del orm, sperm, sperm_ok, orm_ok, s_dict, o_dict

        ##############################################################################################################################################

        ############# 2 Q.ty prescribed exams (Caryo, FRAXA, Other) divided by fam history or No fam history but alterated exams #####################

        if graph_option == "Quantità esami prescritti":

                tot = data[data.tot_prescribed_exams_num != 0].groupby("FamHis_Exams").count()["tot_prescribed_exams_num"].sum()
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.pie(data[data.tot_prescribed_exams_num != 0].groupby("FamHis_Exams").sum()["tot_prescribed_exams_num"],
                       autopct=lambda p: "{:.1f}%\n(N={:.0f})".format(p, p / 100 * tot),
                       labels=data.groupby("FamHis_Exams").sum()["tot_prescribed_exams_num"].index,
                       colors=['darkcyan', 'darkred'], radius=1, labeldistance=1.2,
                       textprops={'size': 'medium', 'weight': 'bold'}, wedgeprops={'edgecolor': 'white'})
                ax.set_title(f"Number of prescribed exams according to criteria\nN_tot_prescribed={tot}", fontsize=15)
                st.pyplot(fig)

        ##############################################################################################################################################

        ############################################## 3 Q.ty "alt" exams / tot ######################################################################

        if graph_option == "Quantità esami alterati":

                data['alt'][data['alt'].notna()] = 1
                data['alt'] = data['alt'].fillna(0)
                data['alt'] = pd.Series(np.where(data.alt.values == 1, "positive_res", 'negative_res'), data.index)
                exams = data[data['completo'] != "no"]
                ### probably needed two nested pie plots - try in chunks
                # first edit df to get col q/ unique exams
                totale_prescritti_pos = exams[exams.alt == "positive_res"]['tot_prescribed_exams_num'].sum()
                exams['Cariotipo'] = pd.Series(np.where(exams.Cariotipo.values == 'si', "Karyotype", ''), exams.index)
                exams['FRAXA'] = pd.Series(np.where(exams.FRAXA.values == 'si', "FRAXA", ''), exams.index)
                exams['Others'] = pd.Series(np.where(exams.Others.values == 'si', "Others", ''), exams.index)
                exams['prescribed_exams'] = exams['Cariotipo'].astype(str) + "-" + exams['FRAXA'].astype(str) + "-" + exams['Others'].astype(str)
                # exams['prescribed_exams'].unique()
                exams.prescribed_exams = [re.sub('--', '-', e) for e in exams.prescribed_exams]
                exams.prescribed_exams = [re.sub(r'^-', '', e) for e in exams.prescribed_exams]
                exams.prescribed_exams = [re.sub(r'-$', '', e) for e in exams.prescribed_exams]
                exams.prescribed_exams.replace('', np.nan, inplace=True)

                fig = plt.figure(figsize=(10.79, 6.075))
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                fig.subplots_adjust(wspace=0)
                explode = [0.05, 0]
                # rotate so that first wedge is split by the x-axis
                angle = 180 * 0.0825
                ax1.pie(exams.groupby("alt").count()["num"], autopct=lambda p: "{:.1f}%\n(N={:.0f})".format(p, p / 100 * exams.groupby("alt").count()["num"].sum()),
                        startangle=angle,
                        labels=exams.groupby("alt").count()["num"].index, explode=explode, textprops={'size': 'medium', 'weight': 'bold'},
                        colors=['darkcyan', 'darkred'],
                        labeldistance=1.05)
                width = .2
                ax2.pie(exams[exams.alt=="positive_res"].groupby('prescribed_exams').count()['num'],
                        autopct=lambda p: "{:.1f}%\n(N={:.0f})".format(p, p / 100 * totale_prescritti_pos) if p > 4 else '',
                        startangle=angle, labels=exams[exams.alt=="positive_res"].groupby('prescribed_exams').count()['num'].index, radius=0.75,
                        textprops={'size': 'smaller', 'weight': 'bold'},
                        labeldistance=1.1)
                n_sub = exams.groupby("alt").count()["num"]['positive_res']
                ax1.set_title(f'Number of positive results in prescribed exams\nN_tot_subject={len(exams)}\nN_tot_exams={exams.tot_prescribed_exams_num.sum()}')
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

                del angle, ax1, ax2, center, con, explode, fig, r, theta1, theta2, width, x, y, n_sub, totale_prescritti_pos

        ##############################################################################################################################################

        #################################### 4 Percentage of DI/DS/MALF etc. by FAM history but examination okay #####################################

        if graph_option == "Percentage of DI/DS/MALF per family history":

                famHis = data[data['FamHis_Exams'] == "FAM_noAlt_exams"]
                col2mod = ['DI/DS/AUT', 'MALF', 'MP', 'PA']
                for c in col2mod:
                        famHis[c].fillna('', inplace=True)

                famHis['FamHistory'] = famHis['DI/DS/AUT'].astype(str) + "+" + famHis['MALF'].astype(str) + "+" + famHis['MP'].astype(str) + "+" + famHis['PA'].astype(str)

                famHis.FamHistory = [e.lstrip('^+') for e in famHis.FamHistory]
                famHis.FamHistory = [e.rstrip(r'+$') for e in famHis.FamHistory]
                famHis.FamHistory = [re.sub(r'\++', '+', e) for e in famHis.FamHistory]

                fig, ax = plt.subplots(figsize=(12, 10))
                wedges, texts, autotexts = ax.pie(famHis.groupby('FamHistory').count().reset_index()['num'],
                                                  autopct=lambda p: "{:.1f}%\n(N={:.0f})".format(p, p / 100 * famHis.groupby('FamHistory').count().reset_index()['num'].sum()),
                                                  labels=famHis.groupby('FamHistory').count().reset_index()['FamHistory'],
                                                  radius=1, labeldistance=1.2,
                                                  textprops={'size': 'medium', 'weight': 'bold'}, wedgeprops={'edgecolor': 'white'})
                ax.set_title(
                        f"Alteration distribution according to family history and normal exams\
                        \nN_tot={famHis.groupby('FamHistory').count().reset_index()['num'].sum()}",
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

                del threshold, pct_label, wedges, autotexts, texts, pct_value, label

        ########################################### 5 number of changes (i.e. "alt" in exams) per couple #############################################

        if graph_option == "Number of alt. exams per couple":
                import math
                couple = data.copy()
                # couple['alter'] = couple[['spermiogramma', 'esami ormonali']].isin(['alt']).any(axis=1)
                # coup_numb = (couple[couple['alter'] == True]).reset_index()
                coup_numb = (couple[couple['esami_alterati'] == 'alt']).reset_index()
                get_index_coup = list(coup_numb['num'])
                couple_df = couple[couple['num'].isin(get_index_coup)]
                no_cahnged_coup = couple[~couple['num'].isin(get_index_coup)]

                couple_df['path'] = 'Path_changed'
                no_cahnged_coup['path'] = 'No_change_in_Path'

                tot_couple = pd.concat([couple_df, no_cahnged_coup], axis=0)

                fig, ax = plt.subplots(figsize=(10, 8))
                ax.pie(tot_couple.groupby('path').count()["num"] / 2,
                       autopct=lambda p: "{:.1f}%\n(N={:.0f})".format(p, p / 100 * len(tot_couple) / 2),
                       labels=tot_couple.groupby('path').count()["num"].index,
                       colors=['darkcyan', 'darkred'], radius=1, labeldistance=1.2,
                       textprops={'size': 'medium', 'weight': 'bold'}, wedgeprops={'edgecolor': 'white'})
                ax.set_title(f"Number of results per couple\nN_tot={math.floor(len(tot_couple) / 2)}", fontsize=15)
                st.pyplot(fig)

                del tot_couple, couple, couple_df, coup_numb, get_index_coup, no_cahnged_coup

        ##############################################################################################################################################

        #################################### 6 N of karyo and FRAXA divided by sex and by sperm/esami ormonali #######################################

        if graph_option == "N of karyo and FRAXA divided by sex and by sperm/esami ormonali":
                left_column, right_column = st.columns(2)
                with left_column:
                        chosen = st.radio('Select variable of interest',
                                          ("All", "Male", "Female", "FRAXA"))
                        # st.write(f"You are in {chosen} house!")
                with right_column:

                        data['alt'][data['alt'].notna()] = 1
                        data['alt'] = data['alt'].fillna(0)
                        data['alt'] = pd.Series(np.where(data.alt.values == 1, "positive_res", 'negative_res'), data.index)

                        ### probably needed two nested pie plots - try in chunks
                        # first edit df to get col q/ unique exams
                        data['Cariotipo'] = pd.Series(np.where(data.Cariotipo.values == 'si', "Karyotype", ''), data.index)
                        data['FRAXA'] = pd.Series(np.where(data.FRAXA.values == 'si', "FRAXA", ''), data.index)
                        data['Others'] = pd.Series(np.where(data.Others.values == 'si', "Others", ''), data.index)
                        data['prescribed_exams'] = data['Cariotipo'].astype(str) + "-" + data['FRAXA'].astype(str) + "-" + data[
                                'Others'].astype(str)
                        # exams['prescribed_exams'].unique()
                        data.prescribed_exams = [re.sub('--', '-', e) for e in data.prescribed_exams]
                        data.prescribed_exams = [re.sub(r'^-', '', e) for e in data.prescribed_exams]
                        data.prescribed_exams = [re.sub(r'-$', '', e) for e in data.prescribed_exams]
                        data.prescribed_exams.replace('', np.nan, inplace=True)

                        exams = data[data['completo'] != "no"]
                        new_ex = exams.copy()
                        new_ex_M = new_ex[new_ex['sex'] == "M"]
                        new_ex_F = new_ex[new_ex['sex'] == "F"]

                        new_ex_M = new_ex_M[new_ex_M['prescribed_exams'] == "Karyotype"]

                        new_ex_karyo = new_ex[new_ex['Cariotipo'] == "Karyotype"]
                        karyo_sunburst = new_ex_karyo.groupby(['FamHis_Exams', 'sex']).count().reset_index()

                        if chosen == "All":
                                fig = px.sunburst(karyo_sunburst, path=["FamHis_Exams", "sex"], values="num", color="FamHis_Exams", names="sex",
                                                  color_discrete_map={'FAM_noAlt_exams': 'darkred', 'noFAM_alt_exams': 'darkcyan'}, width=500,
                                                  height=500)
                                fig.update_traces(textinfo="label+value+percent parent")
                                fig.update_traces(texttemplate='<b>%{label}</b><br><b>N=%{value}</b><br><b>%{percentParent}</b>')
                                fig.update_traces(hovertemplate='<b>%{label}</b><br>N_tot=%{value}<br>', sort=False, rotation=90)
                                fig.update_layout(uniformtext=dict(minsize=5))  # , mode='hide'
                                fig.update_layout(title_font_color="black", title_font_size=20)
                                fig.update_traces(insidetextorientation='horizontal')
                                fig.update_layout(title_text=f'Karyotype examinations<br>\nN_prescribed_karyotype_tot={karyo_sunburst.num.sum()}')
                                st.plotly_chart(fig)

                        elif chosen == "Male":
                                fig, ax = plt.subplots(figsize=(12, 10))
                                ax.pie(karyo_sunburst[karyo_sunburst["sex"] == "M"]['num'],
                                       autopct=lambda p: "{:.1f}%\n(N={:.0f})".format(p, p / 100 * karyo_sunburst[karyo_sunburst["sex"] == "M"][
                                               'num'].sum()),
                                       labels=karyo_sunburst[karyo_sunburst["sex"] == "M"]["FamHis_Exams"],
                                       colors=['darkcyan', 'darkred'], radius=1, labeldistance=1.2,
                                       textprops={'size': 'medium', 'weight': 'bold'}, wedgeprops={'edgecolor': 'white'})
                                ax.set_title(
                                        f"Number of prescribed karyoptype in Male only\nN_tot={karyo_sunburst[karyo_sunburst['sex'] == 'M']['num'].sum()}",
                                        fontsize=15)
                                st.pyplot(fig)
                        elif chosen == "Female":
                                fig, ax = plt.subplots(figsize=(12, 10))
                                ax.pie(karyo_sunburst[karyo_sunburst["sex"] == "F"]['num'],
                                       autopct=lambda p: "{:.1f}%\n(N={:.0f})".format(p, p / 100 * karyo_sunburst[karyo_sunburst["sex"] == "F"][
                                               'num'].sum()),
                                       labels=karyo_sunburst[karyo_sunburst["sex"] == "F"]["FamHis_Exams"],
                                       colors=['darkcyan', 'darkred'], radius=1, labeldistance=1.2,
                                       textprops={'size': 'medium', 'weight': 'bold'}, wedgeprops={'edgecolor': 'white'})
                                ax.set_title(
                                        f"Number of prescribed karyoptype in Female only\nN_tot={karyo_sunburst[karyo_sunburst['sex'] == 'F']['num'].sum()}",
                                        fontsize=15)
                                st.pyplot(fig)
                        else:
                                fraxa = new_ex[new_ex.FRAXA == "FRAXA"]

                                fig, ax = plt.subplots(figsize=(12, 10))
                                ax.pie(fraxa.groupby("FamHis_Exams").count()['num'],
                                       autopct=lambda p: "{:.1f}%\n(N={:.0f})".format(p,
                                                                                      p / 100 * fraxa.groupby("FamHis_Exams").count()['num'].sum()),
                                       labels=fraxa.groupby("FamHis_Exams").count().index,
                                       colors=['darkcyan', 'darkred'], radius=1, labeldistance=1.2,
                                       textprops={'size': 'medium', 'weight': 'bold'}, wedgeprops={'edgecolor': 'white'})
                                ax.set_title(
                                        f"Number of prescribed FRAXA in Female only\nN_tot={fraxa.groupby('FamHis_Exams').count()['num'].sum()}",
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





data_graph = pd.read_excel("C:/Users/Mandich/Desktop/Abstract_cla/my_edit.xlsx", index_col=0)

import matplotlib.pyplot as plt
data_graph= data_graph.transpose().reset_index()

data_graph['inf_perc'] = data_graph['infertilità'] * 100 / (data_graph['infertilità']+data_graph['familiarità'])
data_graph['fam_perc'] = data_graph['familiarità'] * 100 / (data_graph['infertilità']+data_graph['familiarità'])


import matplotlib.patches as mpatches

data_graph['infertilità'] = data_graph['infertilità'].astype(str)
data_graph['familiarità'] = data_graph['familiarità'].astype(str)
data_graph.at[4,'infertilità'] = ''
data_graph.at[8,'infertilità'] = ''
data_graph.at[9,'infertilità'] = ''
data_graph.at[3,'familiarità'] = ''
data_graph.at[6,'familiarità'] = ''
data_graph.at[7,'familiarità'] = ''



ax = data_graph[['index', 'inf_perc','fam_perc']].plot(x='index', kind='bar', stacked=True, color=['#940000','#0179A3'], legend=False, figsize=(6,6))
labels = []
for j in data_graph[['infertilità','familiarità']].columns:
    for i in data_graph.index:
        label = data_graph.loc[i][j]
        labels.append(label)
patches = ax.patches
for label, rect in zip(labels, patches):
    width = rect.get_width()
    if width > 0:
        x = rect.get_x()
        y = rect.get_y()
        height = rect.get_height()
        ax.text(x + width/2., y + height/2., label, ha='center', va='center', fontdict={'fontweight':'bold','color':'white','fontsize':14})
yaxes = ax.axes.get_yaxis()
yaxes.set_visible(False)
plt.xticks(rotation=45, ha='right', fontsize=13)
red_patch = mpatches.Patch(color='#940000', label='Infertilità')
blue_patch = mpatches.Patch(color='#0179A3', label='Familiarità')
plt.legend(handles=[red_patch, blue_patch], loc="right", bbox_to_anchor = (1.1,0.5))
plt.box(False)
plt.xlabel('')
plt.title('ESAMI PRESCRITTI', fontdict={'fontsize':24,'fontweight':'bold'})
plt.show()
