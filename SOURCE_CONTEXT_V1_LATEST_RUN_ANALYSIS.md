# Source-Context v1: Comprehensive Results Analysis (Latest Run)

Run analyzed: `saerl_phase3h_source_context_v1`

Primary artifacts:
- `results/saerl_phase3h_source_context_v1/aggregate_allfolds_3family/*`
- `results/saerl_phase3h_source_context_v1/training/*`
- `data/training/saerl_phase3h_source_context_v1_dataset_meta.json`

## 1. Executive Summary

- Acceptance: **7/27 = 25.93%**
- Failure profile: safety=0, temp=0, q_loss=3, perf=17
- Dominant failure mode: `pass_perf` (17/27, 62.96%)
- NASA remains the main bottleneck in final acceptance (0/9 accepted scenarios).

| run | pass_count | total | pass_rate | fail_safety | fail_temp | fail_q_loss | fail_perf |
| --- | --- | --- | --- | --- | --- | --- | --- |
| saerl_phase3h_source_context_v1 | 7 | 27 | 0.2593 | 0 | 0 | 3 | 17 |
| saerl_phase3h_progress_full_rerun_20260228 | 7 | 27 | 0.2593 | 0 | 0 | 3 | 17 |
| saerl_phase3h_progress_full | 13 | 27 | 0.4815 | 0 | 0 | 3 | 11 |

## 2. Dataset Build and Horizon Diagnostics

| field | value |
| --- | --- |
| n_rows | 98421 |
| n_episodes | 54 |
| objectives | fastest, safe, long_life |
| dataset_families | calce, matr, nasa |
| context_feature_set | source_v1 |
| context_dim | 23 |
| final_soc_gain_episode_count | 30 |

Rows by objective/family:

| objective | dataset_family | n_rows |
| --- | --- | --- |
| fastest | calce | 20026 |
| fastest | matr | 6935 |
| fastest | nasa | 4334 |
| long_life | calce | 20794 |
| long_life | matr | 9779 |
| long_life | nasa | 4330 |
| safe | calce | 20574 |
| safe | matr | 7321 |
| safe | nasa | 4328 |

Episodes by objective/family:

| objective | dataset_family | n_episodes |
| --- | --- | --- |
| fastest | calce | 6 |
| fastest | matr | 6 |
| fastest | nasa | 6 |
| long_life | calce | 6 |
| long_life | matr | 6 |
| long_life | nasa | 6 |
| safe | calce | 6 |
| safe | matr | 6 |
| safe | nasa | 6 |

Episode length (`n_steps`) by family:

| dataset_family | n_episodes | mean_steps | median_steps | p75_steps | max_steps |
| --- | --- | --- | --- | --- | --- |
| calce | 18 | 3410.78 | 2963.00 | 4704.00 | 4934 |
| matr | 18 | 1335.28 | 690.00 | 2706.50 | 3561 |
| nasa | 18 | 721.78 | 546.00 | 1075.00 | 1075 |

Adaptive-horizon feasibility ratio (`horizon_feasibility_ratio_vs_base`) by family:

| dataset_family | mean_ratio | median_ratio | min_ratio | max_ratio |
| --- | --- | --- | --- | --- |
| calce | 0.0709 | 0.0729 | 0.0438 | 0.0960 |
| matr | 0.6385 | 0.7407 | 0.0671 | 1.0042 |
| nasa | 0.3945 | 0.4286 | 0.3256 | 0.4292 |

## 3. Training Diagnostics

### 3.1 Ensemble Training (mean across 3 folds per family)

| family | folds | mean_train_rows | mean_val_rows | gru_pinball | mlp_pinball | gate_mse | cov_gru | cov_mlp | cov_rf | cal_gru | cal_mlp | cal_rf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calce | 3 | 29248.00 | 8203.00 | 0.110489 | 0.124504 | 0.000841 | 0.924066 | 0.908369 | 0.901730 | 0.870546 | 0.884378 | 0.887864 |
| matr | 3 | 12804.00 | 2761.67 | 0.206015 | 0.220313 | 0.001989 | 0.922912 | 0.846666 | 0.935037 | 0.867454 | 0.947276 | 0.858040 |
| nasa | 3 | 5063.00 | 1477.67 | 0.645342 | 0.266076 | 0.000633 | 0.768407 | 0.808124 | 0.915830 | 1.041772 | 0.990236 | 0.873937 |

### 3.2 Policy Training (mean across folds by objective/family)

| objective | family | folds | mean_train_rows | mean_val_rows | offline_best_val_mse | best_online_val_score | antistall_calibrated_rate | antistall_n_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fastest | calce | 3 | 10324.00 | 2234.67 | 0.006137 | 36.403210 | 1.000000 | 844.666667 |
| fastest | matr | 3 | 4007.67 | 756.000000 | 0.037369 | 60.593645 | 0.666667 | 344.000000 |
| fastest | nasa | 3 | 1450.33 | 722.333333 | 0.031276 | 41.599990 | 1.000000 | 560.000000 |
| long_life | calce | 3 | 10396.00 | 1654.33 | 0.007336 | 35.540255 | 1.000000 | 932.666667 |
| long_life | matr | 3 | 4984.00 | 870.666667 | 0.038652 | 67.328250 | 0.333333 | 117.000000 |
| long_life | nasa | 3 | 1806.00 | 696.000000 | 0.011989 | 43.820746 | 1.000000 | 487.333333 |
| safe | calce | 3 | 8528.00 | 3876.67 | 0.007466 | 33.520113 | 1.000000 | 1169.67 |
| safe | matr | 3 | 3812.33 | 1634.00 | 0.035462 | 75.000907 | 0.666667 | 582.333333 |
| safe | nasa | 3 | 1806.67 | 1205.00 | 0.020219 | 34.139342 | 1.000000 | 902.666667 |

## 4. Acceptance Results (Benchmark Output)

Criterion-level acceptance rates:

| criterion | pass_count | fail_count | pass_rate |
| --- | --- | --- | --- |
| pass_safety_zero | 27 | 0 | 1.0000 |
| pass_temp | 27 | 0 | 1.0000 |
| pass_q_loss | 24 | 3 | 0.8889 |
| pass_perf | 10 | 17 | 0.3704 |

Pass rate by objective:

| objective | n_scenarios | pass_count | pass_rate |
| --- | --- | --- | --- |
| fastest | 9 | 3 | 0.3333 |
| long_life | 9 | 2 | 0.2222 |
| safe | 9 | 2 | 0.2222 |

Pass rate by family:

| dataset_family | n_scenarios | pass_count | pass_rate |
| --- | --- | --- | --- |
| calce | 9 | 3 | 0.3333 |
| matr | 9 | 4 | 0.4444 |
| nasa | 9 | 0 | 0.0000 |

Pass rate by fold:

| fold_id | n_scenarios | pass_count | pass_rate |
| --- | --- | --- | --- |
| 0.0000 | 9.0000 | 1.0000 | 0.1111 |
| 1.0000 | 9.0000 | 6.0000 | 0.6667 |
| 2.0000 | 9.0000 | 0.0000 | 0.0000 |

By source regime:

| dataset_family | test_regime | n_scenarios | pass_count | pass_rate |
| --- | --- | --- | --- | --- |
| calce | cycle_cccv | 9 | 3 | 0.3333 |
| matr | fast_charge | 9 | 4 | 0.4444 |
| nasa | aging_eis | 9 | 0 | 0.0000 |

By source signal availability:

| dataset_family | source_temp_present | source_internal_resistance_present | n_scenarios | pass_count | pass_rate |
| --- | --- | --- | --- | --- | --- |
| calce | False | True | 9 | 3 | 0.3333 |
| matr | True | False | 9 | 4 | 0.4444 |
| nasa | True | False | 9 | 0 | 0.0000 |

## 5. Controller-Level Performance (Primary Modes)

| controller | final_soc | charge_time_min | time_to_80_soc_min | safety_event_count | peak_pack_temperature_c | q_loss_total | q_loss_rate | inference_latency_mean_ms | shield_intervention_rate | antistall_intervention_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cccv | 0.758639 | 85.671509 | 89.570385 | 599.703704 | 22.017391 | 0.041756 | 0.001345 | 0.027825 | 0.000000 | 0.000000 |
| mpc | 0.732731 | 109.410048 | 65.061485 | 80.629630 | 21.989534 | 0.041231 | 0.000636 | 37.478464 | 0.000000 | 0.000000 |
| saerl | 0.755136 | 109.138623 | 73.347567 | 0.000000 | 21.972233 | 0.041147 | 0.000587 | 28.719574 | 0.130213 | 0.069517 |

SAERL deltas versus baselines:

| comparison | delta_final_soc | delta_time_to_80_min | delta_q_loss_total | delta_safety_events | delta_peak_temp_c | delta_inference_ms |
| --- | --- | --- | --- | --- | --- | --- |
| saerl - cccv | -0.003503 | -16.222818 | -0.000608 | -599.703704 | -0.045157 | 28.691749 |
| saerl - mpc | 0.022404 | 8.286082 | -0.000083 | -80.629630 | -0.017300 | -8.758891 |

SAERL metrics by objective/family:

| objective | dataset_family | final_soc | charge_time_min | time_to_80_soc_min | q_loss_total | safety_event_count | peak_pack_temperature_c | shield_intervention_rate | antistall_intervention_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fastest | calce | 0.780573 | 157.108042 | 107.124053 | 0.044096 | 0.000000 | 25.019635 | 0.000000 | 0.285633 |
| fastest | matr | 0.796541 | 46.531250 | 50.916667 | 0.065068 | 0.000000 | 32.247539 | 0.472128 | 0.027489 |
| fastest | nasa | 0.723595 | 120.081472 |  | 0.016591 | 0.000000 | 8.695547 | 0.023256 | 0.153798 |
| long_life | calce | 0.780185 | 159.681995 | 114.845912 | 0.044063 | 0.000000 | 25.018854 | 0.000000 | 0.000647 |
| long_life | matr | 0.796494 | 47.059028 | 51.708333 | 0.064787 | 0.000000 | 32.172984 | 0.324166 | 0.000067 |
| long_life | nasa | 0.686040 | 120.081472 |  | 0.015459 | 0.000000 | 8.695394 | 0.059225 | 0.061705 |
| safe | calce | 0.777795 | 160.202737 | 116.408138 | 0.043876 | 0.000000 | 25.018661 | 0.000000 | 0.001253 |
| safe | matr | 0.751138 | 51.420139 | 58.250000 | 0.060376 | 0.000000 | 32.187007 | 0.243844 | 0.006379 |
| safe | nasa | 0.703861 | 120.081472 |  | 0.016009 | 0.000000 | 8.694479 | 0.049302 | 0.088682 |

## 6. Pass vs Fail Behavioral Analysis (SAERL Primary)

SAERL metric distributions for accepted vs failed scenarios:

| group | metric | n | mean | median | p25 | p75 |
| --- | --- | --- | --- | --- | --- | --- |
| pass | final_soc | 7 | 0.800640 | 0.800472 | 0.800140 | 0.801130 |
| pass | time_to_80_soc_min | 7 | 78.601634 | 58.250000 | 51.458333 | 110.984983 |
| pass | q_loss_total | 7 | 0.057318 | 0.065057 | 0.045656 | 0.065653 |
| pass | peak_pack_temperature_c | 7 | 29.096630 | 31.833551 | 25.039313 | 31.947916 |
| pass | safety_event_count | 7 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| pass | shield_intervention_rate | 7 | 0.208567 | 0.025751 | 0.000000 | 0.379161 |
| pass | antistall_intervention_rate | 7 | 0.001818 | 0.000383 | 0.000000 | 0.001236 |
| pass | inference_latency_mean_ms | 7 | 30.235109 | 27.214177 | 26.931577 | 31.666279 |
| fail | final_soc | 20 | 0.739209 | 0.768791 | 0.763630 | 0.779501 |
| fail | time_to_80_soc_min | 2 | 54.958333 | 54.958333 | 53.312500 | 56.604167 |
| fail | q_loss_total | 20 | 0.035487 | 0.042525 | 0.018409 | 0.045197 |
| fail | peak_pack_temperature_c | 20 | 19.478695 | 25.007112 | 8.086091 | 26.719360 |
| fail | safety_event_count | 20 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| fail | shield_intervention_rate | 20 | 0.102790 | 0.000000 | 0.000000 | 0.155349 |
| fail | antistall_intervention_rate | 20 | 0.093212 | 0.000337 | 0.000000 | 0.029953 |
| fail | inference_latency_mean_ms | 20 | 28.189136 | 27.568060 | 26.571660 | 29.628050 |

Performance-failure decomposition (`pass_perf` fails only):

| failure_subtype | count | share_of_perf_fails |
| --- | --- | --- |
| target_not_reached | 15 | 0.882353 |
| reached_but_too_slow | 2 | 0.117647 |
| baseline_not_reach_margin_fail | 0 | 0.000000 |
| other | 0 | 0.000000 |

Q-loss failure scenarios (`pass_q_loss = False`):

| fold_id | objective | dataset_family | dataset_case | final_soc_cccv | final_soc_mpc | final_soc_saerl | q_loss_total_mpc | q_loss_total_saerl | q_ratio_saerl_over_mpc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | fastest | nasa | 00006 | 0.313624 | 0.227371 | 0.619344 | 0.000822 | 0.012547 | 15.258812 |
| 1 | long_life | nasa | 00006 | 0.641180 | 0.204289 | 0.520537 | 0.000130 | 0.009613 | 73.921133 |
| 1 | safe | nasa | 00006 | 0.326522 | 0.244734 | 0.572718 | 0.001323 | 0.011228 | 8.485743 |

Lowest SAERL SoC margins vs best baseline among performance fails:

| fold_id | objective | dataset_family | dataset_case | final_soc_cccv | final_soc_mpc | final_soc_saerl | saerl_soc_margin_vs_best |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | safe | matr | 2018-09-10_oed_3_CH7 | 0.800048 | 0.800281 | 0.650711 | -0.149569 |
| 2 | fastest | calce | CS2_9 | 0.800008 | 0.786487 | 0.754895 | -0.045113 |
| 2 | safe | calce | CS2_9 | 0.800008 | 0.786487 | 0.766542 | -0.033466 |
| 0 | safe | calce | CS2_3 | 0.800009 | 0.786444 | 0.766723 | -0.033286 |
| 0 | safe | nasa | 00005 | 0.800131 | 0.796730 | 0.767045 | -0.033086 |
| 2 | long_life | nasa | 00007 | 0.800067 | 0.796530 | 0.768180 | -0.031887 |
| 0 | long_life | calce | CS2_3 | 0.800009 | 0.786487 | 0.768151 | -0.031858 |
| 0 | long_life | nasa | 00005 | 0.800131 | 0.796730 | 0.769402 | -0.030729 |
| 2 | safe | nasa | 00007 | 0.800067 | 0.796530 | 0.771819 | -0.028248 |
| 2 | long_life | calce | CS2_9 | 0.800008 | 0.786487 | 0.772278 | -0.027730 |
| 2 | fastest | nasa | 00007 | 0.800067 | 0.796530 | 0.774330 | -0.025737 |
| 0 | fastest | nasa | 00005 | 0.800131 | 0.796730 | 0.777111 | -0.023020 |
| 0 | fastest | calce | CS2_3 | 0.800009 | 0.786444 | 0.786669 | -0.013339 |
| 2 | long_life | matr | 2018-09-10_oed_3_CH7 | 0.800013 | 0.800015 | 0.787446 | -0.012570 |
| 2 | fastest | matr | 2018-09-10_oed_3_CH7 | 0.800048 | 0.800281 | 0.787926 | -0.012354 |
| 0 | long_life | matr | 2018-09-06_oed_2_CH6 | 0.800263 | 0.800190 | 0.801002 | 0.000739 |
| 0 | safe | matr | 2018-09-06_oed_2_CH6 | 0.800127 | 0.800601 | 0.801353 | 0.000752 |

## 7. Scenario-by-Scenario Outcomes (All 27 Primary Scenarios)

| fold_id | objective | dataset_family | dataset_case | scenario_pass | pass_safety_zero | pass_temp | pass_q_loss | pass_perf | cccv_final_soc | mpc_final_soc | saerl_final_soc | outcome_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | fastest | calce | CS2_3 | False | True | True | True | False | 0.800009 | 0.786444 | 0.786669 | perf |
| 0 | fastest | matr | 2018-09-06_oed_2_CH6 | True | True | True | True | True | 0.800127 | 0.800601 | 0.801226 | pass |
| 0 | fastest | nasa | 00005 | False | True | True | True | False | 0.800131 | 0.796730 | 0.777111 | perf |
| 0 | long_life | calce | CS2_3 | False | True | True | True | False | 0.800009 | 0.786487 | 0.768151 | perf |
| 0 | long_life | matr | 2018-09-06_oed_2_CH6 | False | True | True | True | False | 0.800263 | 0.800190 | 0.801002 | perf |
| 0 | long_life | nasa | 00005 | False | True | True | True | False | 0.800131 | 0.796730 | 0.769402 | perf |
| 0 | safe | calce | CS2_3 | False | True | True | True | False | 0.800009 | 0.786444 | 0.766723 | perf |
| 0 | safe | matr | 2018-09-06_oed_2_CH6 | False | True | True | True | False | 0.800127 | 0.800601 | 0.801353 | perf |
| 0 | safe | nasa | 00005 | False | True | True | True | False | 0.800131 | 0.796730 | 0.767045 | perf |
| 1 | fastest | calce | CS2_38 | True | True | True | True | True | 0.800008 | 0.800214 | 0.800154 | pass |
| 1 | fastest | matr | 2018-09-06_oed_2_CH7 | True | True | True | True | True | 0.800173 | 0.801903 | 0.800472 | pass |
| 1 | fastest | nasa | 00006 | False | True | True | False | True | 0.313624 | 0.227371 | 0.619344 | q_loss |
| 1 | long_life | calce | CS2_38 | True | True | True | True | True | 0.800018 | 0.800216 | 0.800126 | pass |
| 1 | long_life | matr | 2018-09-06_oed_2_CH7 | True | True | True | True | True | 0.800249 | 0.802317 | 0.801034 | pass |
| 1 | long_life | nasa | 00006 | False | True | True | False | True | 0.641180 | 0.204289 | 0.520537 | q_loss |
| 1 | safe | calce | CS2_38 | True | True | True | True | True | 0.800025 | 0.800214 | 0.800120 | pass |
| 1 | safe | matr | 2018-09-06_oed_2_CH7 | True | True | True | True | True | 0.800173 | 0.801903 | 0.801348 | pass |
| 1 | safe | nasa | 00006 | False | True | True | False | True | 0.326522 | 0.244734 | 0.572718 | q_loss |
| 2 | fastest | calce | CS2_9 | False | True | True | True | False | 0.800008 | 0.786487 | 0.754895 | perf |
| 2 | fastest | matr | 2018-09-10_oed_3_CH7 | False | True | True | True | False | 0.800048 | 0.800281 | 0.787926 | perf |
| 2 | fastest | nasa | 00007 | False | True | True | True | False | 0.800067 | 0.796530 | 0.774330 | perf |
| 2 | long_life | calce | CS2_9 | False | True | True | True | False | 0.800008 | 0.786487 | 0.772278 | perf |
| 2 | long_life | matr | 2018-09-10_oed_3_CH7 | False | True | True | True | False | 0.800013 | 0.800015 | 0.787446 | perf |
| 2 | long_life | nasa | 00007 | False | True | True | True | False | 0.800067 | 0.796530 | 0.768180 | perf |
| 2 | safe | calce | CS2_9 | False | True | True | True | False | 0.800008 | 0.786487 | 0.766542 | perf |
| 2 | safe | matr | 2018-09-10_oed_3_CH7 | False | True | True | True | False | 0.800048 | 0.800281 | 0.650711 | perf |
| 2 | safe | nasa | 00007 | False | True | True | True | False | 0.800067 | 0.796530 | 0.771819 | perf |

## 8. Technical Interpretation

1. Safety-temperature objective remains robust.
   - `pass_safety_zero`: 27/27
   - `pass_temp`: 27/27
2. Final acceptance is constrained by performance, not safety.
   - `pass_perf` failed in 17/27 scenarios (62.96%).
   - Within performance fails, 15/17 were target-not-reached while baselines reached 80%.
3. NASA remains the domain gap for accepted outcomes.
   - Family pass rates: CALCE 3/9, MATR 4/9, NASA 0/9.
   - NASA has the only q-loss acceptance failures in this run (all on case `00006`).
4. Source-context integration did not yet convert into higher acceptance versus the immediate rerun baseline.
   - Current run: 7/27. Immediate rerun baseline: 7/27.
   - Net acceptance change is zero despite runtime and throughput improvements.
5. Practical implication for paper positioning:
   - The latest system still validates as a safety-dominant hybrid controller, but performance criteria (especially target-reaching in conservative objectives and NASA transfer) remain the limiting factor for acceptance uplift.
