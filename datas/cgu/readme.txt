Merge_csv 資料夾說明
1. move-all.csv 存動態的資料，包含跳繩與飛輪
   存放的 header: Note, Tester, Motion, GT(HR), Radar(HR), Power, Distance 
   - 其中 power 為經過人工微調的，會與原來的 power 不太一樣
   - GT(HR) 與 Radar(HR) 在此資料中為 string type, 要使用需要split(",") 再轉為 float 型態
   
2. static-all.csv 為靜態的資料，只有平躺
   存放的 header: Note, Tester, Motion, GT(HR), Radar(HR), Power, Distance 
   其他 0315-0322 的資料在 socio 資料夾中的 other_static
   
3. fitness.csv 為 ipaq 的分數、分類、受試者性別
  存放的 header: Tester, Score, Class, Gender 
   


