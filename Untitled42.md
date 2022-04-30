

```python
import pandas as pd
from pycaret import classification
from pycaret.classification import setup, get_logs,compare_models, create_model, predict_model, tune_model, finalize_model, plot_model, interpret_model, evaluate_model, save_model
from sklearn.ensemble import RandomForestClassifier
```

    C:\Users\lenovo\Anaconda3\lib\site-packages\dask\config.py:131: YAMLLoadWarning: calling yaml.load() without Loader=... is deprecated, as the default Loader is unsafe. Please read https://msg.pyyaml.org/load for full details.
      data = yaml.load(f.read()) or {}
    C:\Users\lenovo\Anaconda3\lib\site-packages\dask\dataframe\utils.py:13: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
      import pandas.util.testing as tm
    C:\Users\lenovo\Anaconda3\lib\site-packages\distributed\config.py:20: YAMLLoadWarning: calling yaml.load() without Loader=... is deprecated, as the default Loader is unsafe. Please read https://msg.pyyaml.org/load for full details.
      defaults = yaml.load(f)
    C:\Users\lenovo\Anaconda3\lib\site-packages\distributed\utils.py:134: RuntimeWarning: Couldn't detect a suitable IP address for reaching '8.8.8.8', defaulting to '127.0.0.1': [WinError 10051] 向一个无法连接的网络尝试了一个套接字操作。
      % (host, default, e), RuntimeWarning)
    


```python
data = pd.read_excel('SJ44.xlsx')
classification_setup = classification.setup(data= data,target='Predictions',session_id=25)
```


<style  type="text/css" >
#T_7c8407cc_c857_11ec_92c9_64bc58becf80row44_col1{
            background-color:  lightgreen;
        }</style><table id="T_7c8407cc_c857_11ec_92c9_64bc58becf80" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Description</th>        <th class="col_heading level0 col1" >Value</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row0_col0" class="data row0 col0" >session_id</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row0_col1" class="data row0 col1" >25</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row1_col0" class="data row1 col0" >Target</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row1_col1" class="data row1 col1" >Predictions</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row2_col0" class="data row2 col0" >Target Type</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row2_col1" class="data row2 col1" >Binary</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row3_col0" class="data row3 col0" >Label Encoded</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row3_col1" class="data row3 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row4_col0" class="data row4 col0" >Original Data</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row4_col1" class="data row4 col1" >(163, 11)</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row5_col0" class="data row5 col0" >Missing Values</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row5_col1" class="data row5 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row6_col0" class="data row6 col0" >Numeric Features</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row6_col1" class="data row6 col1" >9</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row7_col0" class="data row7 col0" >Categorical Features</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row7_col1" class="data row7 col1" >1</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row8_col0" class="data row8 col0" >Ordinal Features</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row8_col1" class="data row8 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row9_col0" class="data row9 col0" >High Cardinality Features</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row9_col1" class="data row9 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row10" class="row_heading level0 row10" >10</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row10_col0" class="data row10 col0" >High Cardinality Method</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row10_col1" class="data row10 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row11" class="row_heading level0 row11" >11</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row11_col0" class="data row11 col0" >Transformed Train Set</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row11_col1" class="data row11 col1" >(114, 10)</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row12" class="row_heading level0 row12" >12</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row12_col0" class="data row12 col0" >Transformed Test Set</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row12_col1" class="data row12 col1" >(49, 10)</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row13" class="row_heading level0 row13" >13</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row13_col0" class="data row13 col0" >Shuffle Train-Test</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row13_col1" class="data row13 col1" >True</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row14" class="row_heading level0 row14" >14</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row14_col0" class="data row14 col0" >Stratify Train-Test</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row14_col1" class="data row14 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row15" class="row_heading level0 row15" >15</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row15_col0" class="data row15 col0" >Fold Generator</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row15_col1" class="data row15 col1" >StratifiedKFold</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row16" class="row_heading level0 row16" >16</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row16_col0" class="data row16 col0" >Fold Number</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row16_col1" class="data row16 col1" >10</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row17" class="row_heading level0 row17" >17</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row17_col0" class="data row17 col0" >CPU Jobs</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row17_col1" class="data row17 col1" >-1</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row18" class="row_heading level0 row18" >18</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row18_col0" class="data row18 col0" >Use GPU</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row18_col1" class="data row18 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row19" class="row_heading level0 row19" >19</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row19_col0" class="data row19 col0" >Log Experiment</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row19_col1" class="data row19 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row20" class="row_heading level0 row20" >20</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row20_col0" class="data row20 col0" >Experiment Name</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row20_col1" class="data row20 col1" >clf-default-name</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row21" class="row_heading level0 row21" >21</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row21_col0" class="data row21 col0" >USI</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row21_col1" class="data row21 col1" >5acc</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row22" class="row_heading level0 row22" >22</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row22_col0" class="data row22 col0" >Imputation Type</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row22_col1" class="data row22 col1" >simple</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row23" class="row_heading level0 row23" >23</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row23_col0" class="data row23 col0" >Iterative Imputation Iteration</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row23_col1" class="data row23 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row24" class="row_heading level0 row24" >24</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row24_col0" class="data row24 col0" >Numeric Imputer</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row24_col1" class="data row24 col1" >mean</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row25" class="row_heading level0 row25" >25</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row25_col0" class="data row25 col0" >Iterative Imputation Numeric Model</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row25_col1" class="data row25 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row26" class="row_heading level0 row26" >26</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row26_col0" class="data row26 col0" >Categorical Imputer</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row26_col1" class="data row26 col1" >constant</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row27" class="row_heading level0 row27" >27</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row27_col0" class="data row27 col0" >Iterative Imputation Categorical Model</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row27_col1" class="data row27 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row28" class="row_heading level0 row28" >28</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row28_col0" class="data row28 col0" >Unknown Categoricals Handling</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row28_col1" class="data row28 col1" >least_frequent</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row29" class="row_heading level0 row29" >29</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row29_col0" class="data row29 col0" >Normalize</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row29_col1" class="data row29 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row30" class="row_heading level0 row30" >30</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row30_col0" class="data row30 col0" >Normalize Method</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row30_col1" class="data row30 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row31" class="row_heading level0 row31" >31</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row31_col0" class="data row31 col0" >Transformation</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row31_col1" class="data row31 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row32" class="row_heading level0 row32" >32</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row32_col0" class="data row32 col0" >Transformation Method</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row32_col1" class="data row32 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row33" class="row_heading level0 row33" >33</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row33_col0" class="data row33 col0" >PCA</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row33_col1" class="data row33 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row34" class="row_heading level0 row34" >34</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row34_col0" class="data row34 col0" >PCA Method</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row34_col1" class="data row34 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row35" class="row_heading level0 row35" >35</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row35_col0" class="data row35 col0" >PCA Components</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row35_col1" class="data row35 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row36" class="row_heading level0 row36" >36</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row36_col0" class="data row36 col0" >Ignore Low Variance</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row36_col1" class="data row36 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row37" class="row_heading level0 row37" >37</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row37_col0" class="data row37 col0" >Combine Rare Levels</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row37_col1" class="data row37 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row38" class="row_heading level0 row38" >38</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row38_col0" class="data row38 col0" >Rare Level Threshold</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row38_col1" class="data row38 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row39" class="row_heading level0 row39" >39</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row39_col0" class="data row39 col0" >Numeric Binning</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row39_col1" class="data row39 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row40" class="row_heading level0 row40" >40</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row40_col0" class="data row40 col0" >Remove Outliers</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row40_col1" class="data row40 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row41" class="row_heading level0 row41" >41</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row41_col0" class="data row41 col0" >Outliers Threshold</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row41_col1" class="data row41 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row42" class="row_heading level0 row42" >42</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row42_col0" class="data row42 col0" >Remove Multicollinearity</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row42_col1" class="data row42 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row43" class="row_heading level0 row43" >43</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row43_col0" class="data row43 col0" >Multicollinearity Threshold</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row43_col1" class="data row43 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row44" class="row_heading level0 row44" >44</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row44_col0" class="data row44 col0" >Remove Perfect Collinearity</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row44_col1" class="data row44 col1" >True</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row45" class="row_heading level0 row45" >45</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row45_col0" class="data row45 col0" >Clustering</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row45_col1" class="data row45 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row46" class="row_heading level0 row46" >46</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row46_col0" class="data row46 col0" >Clustering Iteration</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row46_col1" class="data row46 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row47" class="row_heading level0 row47" >47</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row47_col0" class="data row47 col0" >Polynomial Features</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row47_col1" class="data row47 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row48" class="row_heading level0 row48" >48</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row48_col0" class="data row48 col0" >Polynomial Degree</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row48_col1" class="data row48 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row49" class="row_heading level0 row49" >49</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row49_col0" class="data row49 col0" >Trignometry Features</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row49_col1" class="data row49 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row50" class="row_heading level0 row50" >50</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row50_col0" class="data row50 col0" >Polynomial Threshold</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row50_col1" class="data row50 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row51" class="row_heading level0 row51" >51</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row51_col0" class="data row51 col0" >Group Features</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row51_col1" class="data row51 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row52" class="row_heading level0 row52" >52</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row52_col0" class="data row52 col0" >Feature Selection</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row52_col1" class="data row52 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row53" class="row_heading level0 row53" >53</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row53_col0" class="data row53 col0" >Feature Selection Method</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row53_col1" class="data row53 col1" >classic</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row54" class="row_heading level0 row54" >54</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row54_col0" class="data row54 col0" >Features Selection Threshold</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row54_col1" class="data row54 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row55" class="row_heading level0 row55" >55</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row55_col0" class="data row55 col0" >Feature Interaction</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row55_col1" class="data row55 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row56" class="row_heading level0 row56" >56</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row56_col0" class="data row56 col0" >Feature Ratio</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row56_col1" class="data row56 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row57" class="row_heading level0 row57" >57</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row57_col0" class="data row57 col0" >Interaction Threshold</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row57_col1" class="data row57 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row58" class="row_heading level0 row58" >58</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row58_col0" class="data row58 col0" >Fix Imbalance</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row58_col1" class="data row58 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_7c8407cc_c857_11ec_92c9_64bc58becf80level0_row59" class="row_heading level0 row59" >59</th>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row59_col0" class="data row59 col0" >Fix Imbalance Method</td>
                        <td id="T_7c8407cc_c857_11ec_92c9_64bc58becf80row59_col1" class="data row59 col1" >SMOTE</td>
            </tr>
    </tbody></table>



```python
xgboost_modelz4 = create_model('xgboost') 
```


<style  type="text/css" >
#T_8347aa62_c857_11ec_aa32_64bc58becf80row10_col0,#T_8347aa62_c857_11ec_aa32_64bc58becf80row10_col1,#T_8347aa62_c857_11ec_aa32_64bc58becf80row10_col2,#T_8347aa62_c857_11ec_aa32_64bc58becf80row10_col3,#T_8347aa62_c857_11ec_aa32_64bc58becf80row10_col4,#T_8347aa62_c857_11ec_aa32_64bc58becf80row10_col5,#T_8347aa62_c857_11ec_aa32_64bc58becf80row10_col6{
            background:  yellow;
        }</style><table id="T_8347aa62_c857_11ec_aa32_64bc58becf80" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Accuracy</th>        <th class="col_heading level0 col1" >AUC</th>        <th class="col_heading level0 col2" >Recall</th>        <th class="col_heading level0 col3" >Prec.</th>        <th class="col_heading level0 col4" >F1</th>        <th class="col_heading level0 col5" >Kappa</th>        <th class="col_heading level0 col6" >MCC</th>    </tr>    <tr>        <th class="index_name level0" >Fold</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_8347aa62_c857_11ec_aa32_64bc58becf80level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row0_col0" class="data row0 col0" >0.7500</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row0_col1" class="data row0 col1" >0.9375</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row0_col2" class="data row0 col2" >0.2500</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row0_col3" class="data row0 col3" >1.0000</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row0_col4" class="data row0 col4" >0.4000</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row0_col5" class="data row0 col5" >0.3077</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row0_col6" class="data row0 col6" >0.4264</td>
            </tr>
            <tr>
                        <th id="T_8347aa62_c857_11ec_aa32_64bc58becf80level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row1_col0" class="data row1 col0" >0.9167</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row1_col1" class="data row1 col1" >1.0000</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row1_col2" class="data row1 col2" >0.7500</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row1_col3" class="data row1 col3" >1.0000</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row1_col4" class="data row1 col4" >0.8571</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row1_col5" class="data row1 col5" >0.8000</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row1_col6" class="data row1 col6" >0.8165</td>
            </tr>
            <tr>
                        <th id="T_8347aa62_c857_11ec_aa32_64bc58becf80level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row2_col0" class="data row2 col0" >0.9167</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row2_col1" class="data row2 col1" >0.9688</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row2_col2" class="data row2 col2" >1.0000</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row2_col3" class="data row2 col3" >0.8000</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row2_col4" class="data row2 col4" >0.8889</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row2_col5" class="data row2 col5" >0.8235</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row2_col6" class="data row2 col6" >0.8367</td>
            </tr>
            <tr>
                        <th id="T_8347aa62_c857_11ec_aa32_64bc58becf80level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row3_col0" class="data row3 col0" >0.7500</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row3_col1" class="data row3 col1" >0.8438</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row3_col2" class="data row3 col2" >0.5000</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row3_col3" class="data row3 col3" >0.6667</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row3_col4" class="data row3 col4" >0.5714</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row3_col5" class="data row3 col5" >0.4000</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row3_col6" class="data row3 col6" >0.4082</td>
            </tr>
            <tr>
                        <th id="T_8347aa62_c857_11ec_aa32_64bc58becf80level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row4_col0" class="data row4 col0" >0.7273</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row4_col1" class="data row4 col1" >0.8333</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row4_col2" class="data row4 col2" >0.3333</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row4_col3" class="data row4 col3" >0.5000</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row4_col4" class="data row4 col4" >0.4000</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row4_col5" class="data row4 col5" >0.2326</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row4_col6" class="data row4 col6" >0.2406</td>
            </tr>
            <tr>
                        <th id="T_8347aa62_c857_11ec_aa32_64bc58becf80level0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row5_col0" class="data row5 col0" >0.7273</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row5_col1" class="data row5 col1" >0.8750</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row5_col2" class="data row5 col2" >0.6667</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row5_col3" class="data row5 col3" >0.5000</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row5_col4" class="data row5 col4" >0.5714</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row5_col5" class="data row5 col5" >0.3774</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row5_col6" class="data row5 col6" >0.3858</td>
            </tr>
            <tr>
                        <th id="T_8347aa62_c857_11ec_aa32_64bc58becf80level0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row6_col0" class="data row6 col0" >0.7273</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row6_col1" class="data row6 col1" >0.8750</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row6_col2" class="data row6 col2" >0.6667</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row6_col3" class="data row6 col3" >0.5000</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row6_col4" class="data row6 col4" >0.5714</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row6_col5" class="data row6 col5" >0.3774</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row6_col6" class="data row6 col6" >0.3858</td>
            </tr>
            <tr>
                        <th id="T_8347aa62_c857_11ec_aa32_64bc58becf80level0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row7_col0" class="data row7 col0" >0.9091</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row7_col1" class="data row7 col1" >0.8929</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row7_col2" class="data row7 col2" >0.7500</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row7_col3" class="data row7 col3" >1.0000</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row7_col4" class="data row7 col4" >0.8571</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row7_col5" class="data row7 col5" >0.7925</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row7_col6" class="data row7 col6" >0.8101</td>
            </tr>
            <tr>
                        <th id="T_8347aa62_c857_11ec_aa32_64bc58becf80level0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row8_col0" class="data row8 col0" >0.6364</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row8_col1" class="data row8 col1" >0.7500</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row8_col2" class="data row8 col2" >0.5000</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row8_col3" class="data row8 col3" >0.5000</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row8_col4" class="data row8 col4" >0.5000</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row8_col5" class="data row8 col5" >0.2143</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row8_col6" class="data row8 col6" >0.2143</td>
            </tr>
            <tr>
                        <th id="T_8347aa62_c857_11ec_aa32_64bc58becf80level0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row9_col0" class="data row9 col0" >0.8182</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row9_col1" class="data row9 col1" >0.9286</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row9_col2" class="data row9 col2" >0.7500</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row9_col3" class="data row9 col3" >0.7500</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row9_col4" class="data row9 col4" >0.7500</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row9_col5" class="data row9 col5" >0.6071</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row9_col6" class="data row9 col6" >0.6071</td>
            </tr>
            <tr>
                        <th id="T_8347aa62_c857_11ec_aa32_64bc58becf80level0_row10" class="row_heading level0 row10" >Mean</th>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row10_col0" class="data row10 col0" >0.7879</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row10_col1" class="data row10 col1" >0.8905</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row10_col2" class="data row10 col2" >0.6167</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row10_col3" class="data row10 col3" >0.7217</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row10_col4" class="data row10 col4" >0.6367</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row10_col5" class="data row10 col5" >0.4932</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row10_col6" class="data row10 col6" >0.5131</td>
            </tr>
            <tr>
                        <th id="T_8347aa62_c857_11ec_aa32_64bc58becf80level0_row11" class="row_heading level0 row11" >Std</th>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row11_col0" class="data row11 col0" >0.0925</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row11_col1" class="data row11 col1" >0.0689</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row11_col2" class="data row11 col2" >0.2115</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row11_col3" class="data row11 col3" >0.2095</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row11_col4" class="data row11 col4" >0.1781</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row11_col5" class="data row11 col5" >0.2284</td>
                        <td id="T_8347aa62_c857_11ec_aa32_64bc58becf80row11_col6" class="data row11 col6" >0.2255</td>
            </tr>
    </tbody></table>



```python
tuned_xgboostz4 = tune_model(xgboost_modelz4,optimize = 'AUC',search_library = 'scikit-optimize',search_algorithm = 'bayesian') 
```


<style  type="text/css" >
#T_9b4a50f0_c857_11ec_b64c_64bc58becf80row10_col0,#T_9b4a50f0_c857_11ec_b64c_64bc58becf80row10_col1,#T_9b4a50f0_c857_11ec_b64c_64bc58becf80row10_col2,#T_9b4a50f0_c857_11ec_b64c_64bc58becf80row10_col3,#T_9b4a50f0_c857_11ec_b64c_64bc58becf80row10_col4,#T_9b4a50f0_c857_11ec_b64c_64bc58becf80row10_col5,#T_9b4a50f0_c857_11ec_b64c_64bc58becf80row10_col6{
            background:  yellow;
        }</style><table id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Accuracy</th>        <th class="col_heading level0 col1" >AUC</th>        <th class="col_heading level0 col2" >Recall</th>        <th class="col_heading level0 col3" >Prec.</th>        <th class="col_heading level0 col4" >F1</th>        <th class="col_heading level0 col5" >Kappa</th>        <th class="col_heading level0 col6" >MCC</th>    </tr>    <tr>        <th class="index_name level0" >Fold</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row0_col0" class="data row0 col0" >0.9167</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row0_col1" class="data row0 col1" >0.9062</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row0_col2" class="data row0 col2" >0.7500</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row0_col3" class="data row0 col3" >1.0000</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row0_col4" class="data row0 col4" >0.8571</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row0_col5" class="data row0 col5" >0.8000</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row0_col6" class="data row0 col6" >0.8165</td>
            </tr>
            <tr>
                        <th id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row1_col0" class="data row1 col0" >0.9167</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row1_col1" class="data row1 col1" >1.0000</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row1_col2" class="data row1 col2" >1.0000</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row1_col3" class="data row1 col3" >0.8000</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row1_col4" class="data row1 col4" >0.8889</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row1_col5" class="data row1 col5" >0.8235</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row1_col6" class="data row1 col6" >0.8367</td>
            </tr>
            <tr>
                        <th id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row2_col0" class="data row2 col0" >0.9167</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row2_col1" class="data row2 col1" >0.9375</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row2_col2" class="data row2 col2" >1.0000</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row2_col3" class="data row2 col3" >0.8000</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row2_col4" class="data row2 col4" >0.8889</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row2_col5" class="data row2 col5" >0.8235</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row2_col6" class="data row2 col6" >0.8367</td>
            </tr>
            <tr>
                        <th id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row3_col0" class="data row3 col0" >0.8333</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row3_col1" class="data row3 col1" >0.8750</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row3_col2" class="data row3 col2" >0.5000</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row3_col3" class="data row3 col3" >1.0000</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row3_col4" class="data row3 col4" >0.6667</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row3_col5" class="data row3 col5" >0.5714</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row3_col6" class="data row3 col6" >0.6325</td>
            </tr>
            <tr>
                        <th id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row4_col0" class="data row4 col0" >1.0000</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row4_col1" class="data row4 col1" >1.0000</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row4_col2" class="data row4 col2" >1.0000</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row4_col3" class="data row4 col3" >1.0000</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row4_col4" class="data row4 col4" >1.0000</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row4_col5" class="data row4 col5" >1.0000</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row4_col6" class="data row4 col6" >1.0000</td>
            </tr>
            <tr>
                        <th id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80level0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row5_col0" class="data row5 col0" >0.7273</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row5_col1" class="data row5 col1" >0.7500</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row5_col2" class="data row5 col2" >0.6667</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row5_col3" class="data row5 col3" >0.5000</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row5_col4" class="data row5 col4" >0.5714</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row5_col5" class="data row5 col5" >0.3774</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row5_col6" class="data row5 col6" >0.3858</td>
            </tr>
            <tr>
                        <th id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80level0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row6_col0" class="data row6 col0" >0.8182</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row6_col1" class="data row6 col1" >0.9167</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row6_col2" class="data row6 col2" >1.0000</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row6_col3" class="data row6 col3" >0.6000</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row6_col4" class="data row6 col4" >0.7500</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row6_col5" class="data row6 col5" >0.6207</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row6_col6" class="data row6 col6" >0.6708</td>
            </tr>
            <tr>
                        <th id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80level0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row7_col0" class="data row7 col0" >0.9091</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row7_col1" class="data row7 col1" >1.0000</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row7_col2" class="data row7 col2" >1.0000</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row7_col3" class="data row7 col3" >0.8000</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row7_col4" class="data row7 col4" >0.8889</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row7_col5" class="data row7 col5" >0.8136</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row7_col6" class="data row7 col6" >0.8281</td>
            </tr>
            <tr>
                        <th id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80level0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row8_col0" class="data row8 col0" >0.7273</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row8_col1" class="data row8 col1" >0.7857</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row8_col2" class="data row8 col2" >0.7500</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row8_col3" class="data row8 col3" >0.6000</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row8_col4" class="data row8 col4" >0.6667</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row8_col5" class="data row8 col5" >0.4407</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row8_col6" class="data row8 col6" >0.4485</td>
            </tr>
            <tr>
                        <th id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80level0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row9_col0" class="data row9 col0" >0.7273</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row9_col1" class="data row9 col1" >0.7857</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row9_col2" class="data row9 col2" >0.5000</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row9_col3" class="data row9 col3" >0.6667</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row9_col4" class="data row9 col4" >0.5714</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row9_col5" class="data row9 col5" >0.3774</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row9_col6" class="data row9 col6" >0.3858</td>
            </tr>
            <tr>
                        <th id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80level0_row10" class="row_heading level0 row10" >Mean</th>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row10_col0" class="data row10 col0" >0.8492</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row10_col1" class="data row10 col1" >0.8957</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row10_col2" class="data row10 col2" >0.8167</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row10_col3" class="data row10 col3" >0.7767</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row10_col4" class="data row10 col4" >0.7750</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row10_col5" class="data row10 col5" >0.6648</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row10_col6" class="data row10 col6" >0.6841</td>
            </tr>
            <tr>
                        <th id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80level0_row11" class="row_heading level0 row11" >Std</th>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row11_col0" class="data row11 col0" >0.0926</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row11_col1" class="data row11 col1" >0.0899</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row11_col2" class="data row11 col2" >0.2000</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row11_col3" class="data row11 col3" >0.1739</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row11_col4" class="data row11 col4" >0.1426</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row11_col5" class="data row11 col5" >0.2072</td>
                        <td id="T_9b4a50f0_c857_11ec_b64c_64bc58becf80row11_col6" class="data row11 col6" >0.2052</td>
            </tr>
    </tbody></table>



```python
predict_model(tuned_xgboostz4)
```


<style  type="text/css" >
</style><table id="T_b3c2ac00_c857_11ec_bec0_64bc58becf80" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Model</th>        <th class="col_heading level0 col1" >Accuracy</th>        <th class="col_heading level0 col2" >AUC</th>        <th class="col_heading level0 col3" >Recall</th>        <th class="col_heading level0 col4" >Prec.</th>        <th class="col_heading level0 col5" >F1</th>        <th class="col_heading level0 col6" >Kappa</th>        <th class="col_heading level0 col7" >MCC</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_b3c2ac00_c857_11ec_bec0_64bc58becf80level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_b3c2ac00_c857_11ec_bec0_64bc58becf80row0_col0" class="data row0 col0" >Extreme Gradient Boosting</td>
                        <td id="T_b3c2ac00_c857_11ec_bec0_64bc58becf80row0_col1" class="data row0 col1" >0.8163</td>
                        <td id="T_b3c2ac00_c857_11ec_bec0_64bc58becf80row0_col2" class="data row0 col2" >0.8745</td>
                        <td id="T_b3c2ac00_c857_11ec_bec0_64bc58becf80row0_col3" class="data row0 col3" >0.9333</td>
                        <td id="T_b3c2ac00_c857_11ec_bec0_64bc58becf80row0_col4" class="data row0 col4" >0.6364</td>
                        <td id="T_b3c2ac00_c857_11ec_bec0_64bc58becf80row0_col5" class="data row0 col5" >0.7568</td>
                        <td id="T_b3c2ac00_c857_11ec_bec0_64bc58becf80row0_col6" class="data row0 col6" >0.6175</td>
                        <td id="T_b3c2ac00_c857_11ec_bec0_64bc58becf80row0_col7" class="data row0 col7" >0.6468</td>
            </tr>
    </tbody></table>





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>eGFR</th>
      <th>NIHSS</th>
      <th>Albumin</th>
      <th>Albumin-to-globulin_ratio</th>
      <th>Serum_creatinine</th>
      <th>Blood_neutrophils_count</th>
      <th>Blood_neutrophils_count.1</th>
      <th>Fasting_blood_glucose</th>
      <th>Collateral_status_0</th>
      <th>Predictions</th>
      <th>Label</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>60.0</td>
      <td>107.900002</td>
      <td>20.0</td>
      <td>37.700001</td>
      <td>1.33</td>
      <td>69.000000</td>
      <td>8.700000</td>
      <td>7.00</td>
      <td>5.07</td>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>0.5231</td>
    </tr>
    <tr>
      <th>1</th>
      <td>62.0</td>
      <td>155.399994</td>
      <td>9.0</td>
      <td>40.799999</td>
      <td>1.28</td>
      <td>50.000000</td>
      <td>7.500000</td>
      <td>6.20</td>
      <td>5.38</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>0.6382</td>
    </tr>
    <tr>
      <th>2</th>
      <td>61.0</td>
      <td>105.800003</td>
      <td>15.0</td>
      <td>35.799999</td>
      <td>1.18</td>
      <td>54.000000</td>
      <td>10.900000</td>
      <td>7.90</td>
      <td>7.71</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.5191</td>
    </tr>
    <tr>
      <th>3</th>
      <td>79.0</td>
      <td>74.099998</td>
      <td>17.0</td>
      <td>35.700001</td>
      <td>1.29</td>
      <td>91.000000</td>
      <td>9.500000</td>
      <td>7.80</td>
      <td>5.92</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.6562</td>
    </tr>
    <tr>
      <th>4</th>
      <td>76.0</td>
      <td>66.199997</td>
      <td>15.0</td>
      <td>37.400002</td>
      <td>1.80</td>
      <td>101.000000</td>
      <td>9.640000</td>
      <td>8.05</td>
      <td>6.39</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.5948</td>
    </tr>
    <tr>
      <th>5</th>
      <td>61.0</td>
      <td>99.199997</td>
      <td>12.0</td>
      <td>40.799999</td>
      <td>1.56</td>
      <td>74.000000</td>
      <td>10.500000</td>
      <td>8.30</td>
      <td>5.26</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>0.6743</td>
    </tr>
    <tr>
      <th>6</th>
      <td>65.0</td>
      <td>102.300003</td>
      <td>38.0</td>
      <td>37.400002</td>
      <td>1.29</td>
      <td>55.000000</td>
      <td>20.100000</td>
      <td>18.40</td>
      <td>14.40</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.7250</td>
    </tr>
    <tr>
      <th>7</th>
      <td>56.0</td>
      <td>117.199997</td>
      <td>16.0</td>
      <td>43.900002</td>
      <td>1.60</td>
      <td>65.000000</td>
      <td>11.600000</td>
      <td>9.70</td>
      <td>5.48</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>0.6299</td>
    </tr>
    <tr>
      <th>8</th>
      <td>47.0</td>
      <td>102.900002</td>
      <td>11.0</td>
      <td>39.799999</td>
      <td>2.06</td>
      <td>75.000000</td>
      <td>8.800000</td>
      <td>7.20</td>
      <td>5.93</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>0.7280</td>
    </tr>
    <tr>
      <th>9</th>
      <td>82.0</td>
      <td>86.599998</td>
      <td>13.0</td>
      <td>40.099998</td>
      <td>1.14</td>
      <td>79.000000</td>
      <td>10.700000</td>
      <td>8.80</td>
      <td>5.14</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.5704</td>
    </tr>
    <tr>
      <th>10</th>
      <td>78.0</td>
      <td>81.500000</td>
      <td>17.0</td>
      <td>36.200001</td>
      <td>1.27</td>
      <td>84.000000</td>
      <td>8.700000</td>
      <td>7.60</td>
      <td>6.63</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.6688</td>
    </tr>
    <tr>
      <th>11</th>
      <td>73.0</td>
      <td>61.799999</td>
      <td>19.0</td>
      <td>35.500000</td>
      <td>1.45</td>
      <td>108.000000</td>
      <td>7.300000</td>
      <td>6.00</td>
      <td>9.29</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.7349</td>
    </tr>
    <tr>
      <th>12</th>
      <td>64.0</td>
      <td>104.699997</td>
      <td>12.0</td>
      <td>41.700001</td>
      <td>2.16</td>
      <td>70.000000</td>
      <td>11.600000</td>
      <td>10.50</td>
      <td>5.22</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>0.5682</td>
    </tr>
    <tr>
      <th>13</th>
      <td>71.0</td>
      <td>103.000000</td>
      <td>13.0</td>
      <td>37.500000</td>
      <td>1.38</td>
      <td>70.000000</td>
      <td>7.300000</td>
      <td>5.50</td>
      <td>6.55</td>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>0.5124</td>
    </tr>
    <tr>
      <th>14</th>
      <td>64.0</td>
      <td>92.400002</td>
      <td>25.0</td>
      <td>39.900002</td>
      <td>1.38</td>
      <td>78.000000</td>
      <td>11.000000</td>
      <td>9.70</td>
      <td>12.35</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.6867</td>
    </tr>
    <tr>
      <th>15</th>
      <td>84.0</td>
      <td>122.400002</td>
      <td>10.0</td>
      <td>37.599998</td>
      <td>1.65</td>
      <td>45.000000</td>
      <td>8.900000</td>
      <td>7.80</td>
      <td>6.94</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>0.7022</td>
    </tr>
    <tr>
      <th>16</th>
      <td>68.0</td>
      <td>103.400002</td>
      <td>14.0</td>
      <td>38.700001</td>
      <td>1.35</td>
      <td>70.000000</td>
      <td>13.000000</td>
      <td>10.90</td>
      <td>6.47</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.5746</td>
    </tr>
    <tr>
      <th>17</th>
      <td>77.0</td>
      <td>94.800003</td>
      <td>11.0</td>
      <td>34.599998</td>
      <td>1.08</td>
      <td>57.000000</td>
      <td>6.900000</td>
      <td>5.40</td>
      <td>5.80</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.6069</td>
    </tr>
    <tr>
      <th>18</th>
      <td>84.0</td>
      <td>91.500000</td>
      <td>16.0</td>
      <td>34.900002</td>
      <td>1.77</td>
      <td>75.000000</td>
      <td>4.800000</td>
      <td>3.60</td>
      <td>6.76</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.6487</td>
    </tr>
    <tr>
      <th>19</th>
      <td>71.0</td>
      <td>107.199997</td>
      <td>17.0</td>
      <td>36.400002</td>
      <td>1.55</td>
      <td>52.000000</td>
      <td>10.300000</td>
      <td>8.40</td>
      <td>6.75</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>0.5600</td>
    </tr>
    <tr>
      <th>20</th>
      <td>78.0</td>
      <td>87.900002</td>
      <td>16.0</td>
      <td>40.700001</td>
      <td>1.57</td>
      <td>50.000000</td>
      <td>9.700000</td>
      <td>7.90</td>
      <td>7.91</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.5346</td>
    </tr>
    <tr>
      <th>21</th>
      <td>69.0</td>
      <td>57.000000</td>
      <td>20.0</td>
      <td>35.599998</td>
      <td>1.44</td>
      <td>117.000000</td>
      <td>6.600000</td>
      <td>4.70</td>
      <td>4.94</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.6594</td>
    </tr>
    <tr>
      <th>22</th>
      <td>87.0</td>
      <td>100.599998</td>
      <td>8.0</td>
      <td>36.799999</td>
      <td>1.82</td>
      <td>53.000000</td>
      <td>6.700000</td>
      <td>5.50</td>
      <td>4.69</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>0.7465</td>
    </tr>
    <tr>
      <th>23</th>
      <td>70.0</td>
      <td>102.900002</td>
      <td>11.0</td>
      <td>39.099998</td>
      <td>1.48</td>
      <td>54.000000</td>
      <td>11.100000</td>
      <td>9.70</td>
      <td>11.95</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.5008</td>
    </tr>
    <tr>
      <th>24</th>
      <td>64.0</td>
      <td>110.099998</td>
      <td>12.0</td>
      <td>38.000000</td>
      <td>1.51</td>
      <td>67.000000</td>
      <td>10.600000</td>
      <td>8.60</td>
      <td>14.60</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.5628</td>
    </tr>
    <tr>
      <th>25</th>
      <td>78.0</td>
      <td>41.700001</td>
      <td>18.0</td>
      <td>39.500000</td>
      <td>1.69</td>
      <td>116.000000</td>
      <td>7.400000</td>
      <td>6.00</td>
      <td>5.66</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.5396</td>
    </tr>
    <tr>
      <th>26</th>
      <td>79.0</td>
      <td>60.299999</td>
      <td>15.0</td>
      <td>37.299999</td>
      <td>1.48</td>
      <td>84.000000</td>
      <td>7.100000</td>
      <td>6.40</td>
      <td>8.52</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.6219</td>
    </tr>
    <tr>
      <th>27</th>
      <td>78.0</td>
      <td>124.900002</td>
      <td>12.0</td>
      <td>33.900002</td>
      <td>1.86</td>
      <td>58.000000</td>
      <td>6.000000</td>
      <td>4.80</td>
      <td>5.60</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>0.5161</td>
    </tr>
    <tr>
      <th>28</th>
      <td>51.0</td>
      <td>117.300003</td>
      <td>18.0</td>
      <td>37.900002</td>
      <td>1.64</td>
      <td>66.000000</td>
      <td>5.100000</td>
      <td>4.00</td>
      <td>6.72</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>0.6743</td>
    </tr>
    <tr>
      <th>29</th>
      <td>80.0</td>
      <td>66.500000</td>
      <td>12.0</td>
      <td>41.599998</td>
      <td>1.83</td>
      <td>77.000000</td>
      <td>16.299999</td>
      <td>14.60</td>
      <td>9.18</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.6940</td>
    </tr>
    <tr>
      <th>30</th>
      <td>66.0</td>
      <td>131.899994</td>
      <td>20.0</td>
      <td>38.000000</td>
      <td>1.76</td>
      <td>44.000000</td>
      <td>4.800000</td>
      <td>3.70</td>
      <td>6.44</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>0.5361</td>
    </tr>
    <tr>
      <th>31</th>
      <td>57.0</td>
      <td>125.599998</td>
      <td>21.0</td>
      <td>38.299999</td>
      <td>1.54</td>
      <td>61.000000</td>
      <td>9.800000</td>
      <td>8.00</td>
      <td>6.46</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.5005</td>
    </tr>
    <tr>
      <th>32</th>
      <td>81.0</td>
      <td>92.000000</td>
      <td>10.0</td>
      <td>37.799999</td>
      <td>1.72</td>
      <td>72.000000</td>
      <td>7.500000</td>
      <td>6.00</td>
      <td>9.98</td>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>0.5695</td>
    </tr>
    <tr>
      <th>33</th>
      <td>62.0</td>
      <td>105.500000</td>
      <td>8.0</td>
      <td>39.200001</td>
      <td>1.57</td>
      <td>54.000000</td>
      <td>6.300000</td>
      <td>5.70</td>
      <td>6.31</td>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>0.6978</td>
    </tr>
    <tr>
      <th>34</th>
      <td>58.0</td>
      <td>91.599998</td>
      <td>10.0</td>
      <td>43.000000</td>
      <td>1.80</td>
      <td>80.000000</td>
      <td>9.200000</td>
      <td>7.70</td>
      <td>7.44</td>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>0.6211</td>
    </tr>
    <tr>
      <th>35</th>
      <td>72.0</td>
      <td>96.099998</td>
      <td>13.0</td>
      <td>42.099998</td>
      <td>1.81</td>
      <td>57.000000</td>
      <td>11.900000</td>
      <td>11.00</td>
      <td>7.79</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.6491</td>
    </tr>
    <tr>
      <th>36</th>
      <td>56.0</td>
      <td>89.599998</td>
      <td>11.0</td>
      <td>35.700001</td>
      <td>1.69</td>
      <td>82.000000</td>
      <td>8.900000</td>
      <td>6.30</td>
      <td>4.57</td>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>0.5719</td>
    </tr>
    <tr>
      <th>37</th>
      <td>78.0</td>
      <td>68.900002</td>
      <td>6.0</td>
      <td>37.500000</td>
      <td>1.18</td>
      <td>75.000000</td>
      <td>5.000000</td>
      <td>3.50</td>
      <td>5.48</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>0.6143</td>
    </tr>
    <tr>
      <th>38</th>
      <td>73.0</td>
      <td>87.000000</td>
      <td>12.0</td>
      <td>39.299999</td>
      <td>1.62</td>
      <td>62.000000</td>
      <td>8.000000</td>
      <td>6.60</td>
      <td>7.35</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0.5256</td>
    </tr>
    <tr>
      <th>39</th>
      <td>86.0</td>
      <td>111.300003</td>
      <td>14.0</td>
      <td>35.099998</td>
      <td>1.54</td>
      <td>63.000000</td>
      <td>8.800000</td>
      <td>6.20</td>
      <td>4.53</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.5187</td>
    </tr>
    <tr>
      <th>40</th>
      <td>65.0</td>
      <td>104.500000</td>
      <td>4.0</td>
      <td>43.500000</td>
      <td>1.89</td>
      <td>54.000000</td>
      <td>8.600000</td>
      <td>7.40</td>
      <td>6.42</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0.7266</td>
    </tr>
    <tr>
      <th>41</th>
      <td>70.0</td>
      <td>93.720001</td>
      <td>7.0</td>
      <td>37.639999</td>
      <td>1.56</td>
      <td>72.989998</td>
      <td>8.700000</td>
      <td>6.90</td>
      <td>5.62</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0.7574</td>
    </tr>
    <tr>
      <th>42</th>
      <td>70.0</td>
      <td>145.500000</td>
      <td>14.0</td>
      <td>40.700001</td>
      <td>1.55</td>
      <td>40.000000</td>
      <td>9.900000</td>
      <td>7.70</td>
      <td>6.45</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>0.5871</td>
    </tr>
    <tr>
      <th>43</th>
      <td>66.0</td>
      <td>126.699997</td>
      <td>24.0</td>
      <td>35.099998</td>
      <td>1.87</td>
      <td>59.000000</td>
      <td>16.100000</td>
      <td>14.90</td>
      <td>7.16</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.6772</td>
    </tr>
    <tr>
      <th>44</th>
      <td>69.0</td>
      <td>137.899994</td>
      <td>16.0</td>
      <td>42.799999</td>
      <td>1.38</td>
      <td>42.000000</td>
      <td>11.700000</td>
      <td>9.00</td>
      <td>8.13</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.5246</td>
    </tr>
    <tr>
      <th>45</th>
      <td>82.0</td>
      <td>91.800003</td>
      <td>20.0</td>
      <td>34.200001</td>
      <td>1.40</td>
      <td>61.000000</td>
      <td>12.400000</td>
      <td>10.40</td>
      <td>5.79</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.6740</td>
    </tr>
    <tr>
      <th>46</th>
      <td>70.0</td>
      <td>96.699997</td>
      <td>17.0</td>
      <td>37.639999</td>
      <td>1.56</td>
      <td>54.000000</td>
      <td>7.500000</td>
      <td>6.20</td>
      <td>4.78</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>0.7091</td>
    </tr>
    <tr>
      <th>47</th>
      <td>66.0</td>
      <td>119.699997</td>
      <td>20.0</td>
      <td>38.000000</td>
      <td>1.32</td>
      <td>62.000000</td>
      <td>11.600000</td>
      <td>8.80</td>
      <td>4.54</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.5408</td>
    </tr>
    <tr>
      <th>48</th>
      <td>93.0</td>
      <td>64.300003</td>
      <td>14.0</td>
      <td>34.099998</td>
      <td>1.54</td>
      <td>100.000000</td>
      <td>7.700000</td>
      <td>7.20</td>
      <td>7.35</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.7404</td>
    </tr>
  </tbody>
</table>
</div>




```python
from pycaret.regression import *

X_train = get_config('X_train') 

X_test = get_config('X_test') 

y_train = get_config('y_train') 

y_test = get_config('y_test') 

X = data.drop(columns='Predictions',axis=1)

import pickle

model = pickle.dump(tuned_xgboostz4,open('model.p','wb'))

model_rf = pickle.dump(tuned_xgboostz4,open('tuned_xgboostz4.p','wb'))

save_model(tuned_xgboostz4,'tuned_xgboostz4')
```

    Transformation Pipeline and Model Successfully Saved
    




    (Pipeline(memory=None,
              steps=[('dtypes',
                      DataTypes_Auto_infer(categorical_features=[],
                                           display_types=True, features_todrop=[],
                                           id_columns=[],
                                           ml_usecase='classification',
                                           numerical_features=[],
                                           target='Predictions', time_features=[])),
                     ('imputer',
                      Simple_Imputer(categorical_strategy='not_available',
                                     fill_value_categorical=None,
                                     fill_value_numerical=None,
                                     numeric_...
                                    monotone_constraints='()', n_estimators=62,
                                    n_jobs=-1, num_parallel_tree=1,
                                    objective='binary:logistic', predictor='auto',
                                    random_state=25, reg_alpha=6.653422504098605e-10,
                                    reg_lambda=1.8943114418852933e-07,
                                    scale_pos_weight=5.568525342266825,
                                    subsample=0.5984971208724539, tree_method='auto',
                                    use_label_encoder=True, validate_parameters=1,
                                    verbosity=0)]],
              verbose=False),
     'tuned_xgboostz4.pkl')




```python
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

pipeline = get_config('prep_pipe')

pipeline = pickle.dump(tuned_xgboostz4,open('pipeline.p','wb'))

pipe = get_config('prep_pipe')

pipeline = pickle.dump(pipe,open('pipeline.p','wb'))

pickle.load(open( "model.p", "rb" ))

model = pickle.load(open( "model.p", "rb" ))

pipelines = pickle.load(open( "pipeline.p", "rb" ))

print(pipelines)

pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])

pred = tuned_xgboostz4.predict(X_test)

pred
```

    Pipeline(memory=None,
             steps=[('dtypes',
                     DataTypes_Auto_infer(categorical_features=[],
                                          display_types=True, features_todrop=[],
                                          id_columns=[],
                                          ml_usecase='classification',
                                          numerical_features=[],
                                          target='Predictions', time_features=[])),
                    ('imputer',
                     Simple_Imputer(categorical_strategy='not_available',
                                    fill_value_categorical=None,
                                    fill_value_numerical=None,
                                    numeric_...
                    ('scaling', 'passthrough'), ('P_transform', 'passthrough'),
                    ('binn', 'passthrough'), ('rem_outliers', 'passthrough'),
                    ('cluster_all', 'passthrough'),
                    ('dummy', Dummify(target='Predictions')),
                    ('fix_perfect', Remove_100(target='Predictions')),
                    ('clean_names', Clean_Colum_Names()),
                    ('feature_select', 'passthrough'), ('fix_multi', 'passthrough'),
                    ('dfs', 'passthrough'), ('pca', 'passthrough')],
             verbose=False)
    




    array([1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0,
           1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0,
           0, 0, 1, 0, 0], dtype=int64)




```python
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown='ignore')

one_hot_encode_cols = X.dtypes[X.dtypes == np.object]  

one_hot_encode_cols = one_hot_encode_cols.index.tolist() 

X[one_hot_encode_cols].head().T

df = pd.get_dummies(X, columns=['Collateral_status'])

df.describe().T

df['Collateral_status']=pd.factorize(X['Collateral_status'])[0]

df1 = X
```


```python
enc_df = pd.DataFrame(enc.fit_transform(X[['Collateral_status']]).toarray())
```


```python
df1 = X

df1['Age'] = X['Age']

df1['NIHSS'] = X['NIHSS']

df1['Blood_neutrophils_count'] = X['Blood_neutrophils_count']

df1['Fasting_blood_glucose'] = X['Fasting_blood_glucose']

df1['White_blood_cell_count'] = X['White_blood_cell_count']

df1['Serum_creatinine'] = X['Serum_creatinine']

df1['Albumin'] = X['Albumin']

df1['Albumin-to-globulin_ratio'] = X['Albumin-to-globulin_ratio']

df1['eGFR'] = X['eGFR']

df1['Collateral_status'] = pd.factorize(X['Collateral_status'])[0]

df1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>eGFR</th>
      <th>Age</th>
      <th>NIHSS</th>
      <th>Albumin</th>
      <th>Albumin-to-globulin_ratio</th>
      <th>Serum_creatinine</th>
      <th>White_blood_cell_count</th>
      <th>Blood_neutrophils_count</th>
      <th>Fasting_blood_glucose</th>
      <th>Collateral_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>118.1</td>
      <td>81</td>
      <td>5</td>
      <td>36.7</td>
      <td>1.96</td>
      <td>49.0</td>
      <td>8.9</td>
      <td>6.7</td>
      <td>5.16</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37.6</td>
      <td>74</td>
      <td>20</td>
      <td>35.2</td>
      <td>1.40</td>
      <td>128.0</td>
      <td>10.5</td>
      <td>8.5</td>
      <td>16.25</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>43.2</td>
      <td>76</td>
      <td>19</td>
      <td>40.5</td>
      <td>1.53</td>
      <td>113.0</td>
      <td>13.1</td>
      <td>11.6</td>
      <td>8.60</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>68.0</td>
      <td>83</td>
      <td>18</td>
      <td>41.2</td>
      <td>1.36</td>
      <td>75.0</td>
      <td>10.4</td>
      <td>9.6</td>
      <td>7.48</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>77.6</td>
      <td>76</td>
      <td>20</td>
      <td>37.3</td>
      <td>1.36</td>
      <td>88.0</td>
      <td>15.1</td>
      <td>14.0</td>
      <td>8.94</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>158</th>
      <td>160.4</td>
      <td>50</td>
      <td>14</td>
      <td>39.5</td>
      <td>1.49</td>
      <td>39.0</td>
      <td>9.4</td>
      <td>6.9</td>
      <td>4.70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>159</th>
      <td>142.3</td>
      <td>68</td>
      <td>33</td>
      <td>39.9</td>
      <td>1.49</td>
      <td>41.0</td>
      <td>11.1</td>
      <td>10.0</td>
      <td>7.09</td>
      <td>0</td>
    </tr>
    <tr>
      <th>160</th>
      <td>81.5</td>
      <td>78</td>
      <td>17</td>
      <td>36.2</td>
      <td>1.27</td>
      <td>84.0</td>
      <td>8.7</td>
      <td>7.6</td>
      <td>6.63</td>
      <td>1</td>
    </tr>
    <tr>
      <th>161</th>
      <td>105.1</td>
      <td>63</td>
      <td>21</td>
      <td>39.6</td>
      <td>2.26</td>
      <td>54.0</td>
      <td>7.5</td>
      <td>6.6</td>
      <td>5.51</td>
      <td>0</td>
    </tr>
    <tr>
      <th>162</th>
      <td>129.6</td>
      <td>72</td>
      <td>20</td>
      <td>37.7</td>
      <td>2.20</td>
      <td>44.0</td>
      <td>6.1</td>
      <td>5.1</td>
      <td>6.38</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>163 rows × 10 columns</p>
</div>




```python
pd.DataFrame(model.predict(X_test))

tuned_xgboostz4.fit(X_train,y_train)

fit_model = pickle.dump(tuned_xgboostz4,open('tuned_xgboostz4.p','wb'))

pred = tuned_xgboostz4.predict(X_test)
```


```python

```
