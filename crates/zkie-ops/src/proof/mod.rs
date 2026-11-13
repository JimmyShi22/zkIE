//! Relation proof interfaces and shared model commitment.
pub mod base_final_session;
pub mod base_last_session;
pub mod bat_fold_session;
pub mod binding;
pub mod column_session;
pub mod committed_psc;
pub mod cs_final_session;
pub mod cs_multi_session;
pub mod edge_session;
pub mod fold_handoff;
pub mod group_commitment_session;
pub mod group_row_session;
pub mod gu_combine_session;
pub mod hidden_send_session;
pub mod hidden_norm;
pub mod hidden_norm_psc;
pub mod send_session;
pub mod lin_session;
pub mod output_head_lin;
pub mod output_closure;
pub mod plan;
pub mod proximity_session;
pub mod psc;
pub mod psc_general;
pub mod lookup;
pub mod lookup_messages;
pub mod lookup_table;
pub mod relations;
pub mod session_baseline;
pub mod multi_stage_iopp;
pub mod proximity_iopp;
pub mod interval;
pub mod proximity_planner;
pub mod model_commitment;
pub mod norm_param_table;

#[cfg(test)]
mod joint_view_tests;
