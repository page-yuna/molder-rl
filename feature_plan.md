# Cycle-Level Feature Plan (Molding-QA dataset)

| Feature name (column)       | Source column(s)                           | Rationale / what it captures                          |
|-----------------------------|--------------------------------------------|-------------------------------------------------------|
| cycle_time                  | `Cycle_Time`                               | Overall productivity KPI                              |
| filling_time                | `Filling_Time`                             | Long fill → short-shot risk                           |
| plasticizing_time           | `Plasticizing_Time`                        | Material melt quality                                 |
| clamp_close_time            | `Clamp_Close_Time`                         | Slow clamp may cause flash                            |
| max_injection_pressure      | `Max_Injection_Pressure`                   | Peak cavity/ram pressure → flash / stress             |
| max_back_pressure           | `Max_Back_Pressure`                        | Excess melt back pressure → overheating               |
| avg_back_pressure           | `Average_Back_Pressure`                    | Long-term drift indicator                             |
| max_screw_rpm               | `Max_Screw_RPM`                            | Extreme RPM unstable material feed                    |
| avg_screw_rpm               | `Average_Screw_RPM`                        | Melt homogeneity trend                                |
| max_injection_speed         | `Max_Injection_Speed`                      | Fast + high pressure ⇒ flash; too slow ⇒ short-shot   |
| cushion_error               | `Cushion_Position` − target (5 mm)         | Shot size consistency                                 |
| switch_over_error           | `Switch_Over_Position` − set-point (15 mm) | Fill/pack switchover accuracy                         |
| barrel_temp_avg             | mean(`Barrel_Temperature_1…7`)             | Material viscosity proxy                              |
| hopper_temp                 | `Hopper_Temperature`                       | Moisture control                                      |
| mold_temp_avg               | mean(`Mold_Temperature_1…12`)              | Surface finish & cycle driver                         |
| delta_mold_temp             | max − min of 12 mold temps                 | Gradient → warpage risk                               |
| outlier_flag                | from `outlier_flag.py`                     | IQR anomaly indicator                                 |
| label                       | `label` (0 = pass, 1 = fail)               | Supervised target & RL penalty                        |
