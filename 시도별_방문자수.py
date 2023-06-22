import pandas as pd
pd.set_option('mode.chained_assignment',  None) # Warning 방지용
import numpy as np

data = pd.read_csv('./DATA/시도별_방문자수.csv', index_col=0, encoding='CP949', engine='python')
data.head() 

data.columns


#바 차트 그리기
from matplotlib import pyplot as plt
from matplotlib import rcParams, style
style.use('ggplot')

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

MC_ratio = data[['방문자수']]
MC_ratio = MC_ratio.sort_values('방문자수', ascending = False)
plt.rcParams["figure.figsize"] = (25,5)
MC_ratio.plot(kind='bar', rot=90)
plt.show()



MC_ratio = data[['전년도 방문자수']]
MC_ratio = MC_ratio.sort_values('전년도 방문자수', ascending = False)
plt.rcParams["figure.figsize"] = (25,5)
MC_ratio.plot(kind='bar', rot=90)
plt.show()


