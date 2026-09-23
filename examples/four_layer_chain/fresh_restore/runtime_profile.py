"""Finite whole-phase budget; no resource admission is granted by this file."""
from weight_upload import schedule


def profiles(host, mode, nested_source_bytes):
    budget = host['baseline_budget' if mode == 'baseline' else 'restored_budget']
    groups = schedule([i['prepared'] for i in host['copies']['uploads'] if i['symbol'] == 'weights'])
    runtime = dict(startup_seconds=840, startup_pre_start_seconds=240, startup_device_seconds=600,
                   capture_seconds=600, normal_stop_seconds=30, runtime_seconds=1500,
                   compute_observer_seconds=90, observer_poll_interval_seconds=.5,
                   capture_progress_seconds=15, parent_poll_seconds=.1, cpu=[0],
                   address_space_bytes=8 << 30, RSS_bytes=1 << 30, guard_RSS_bytes=256 << 20, swap_bytes=0,
                   max_journal_bytes=128 << 20, max_events=160000 if mode == 'baseline' else 80000,
                   capture_directory_bytes=1 << 30 if mode == 'baseline' else 512 << 20,
                   prepared_directory_bytes=4 << 30, prepared_cache_bytes=16 << 20,
                   max_raw_bytes=budget['raw_byte_ceiling'], max_raw_files=budget['raw_file_ceiling'],
                   worker_cpu_millicores=1000, worker_memory_bytes=8 << 30,
                   host_log_bytes=8 << 20, host_disk_reserve_bytes=32 << 30, host_available_memory_reserve_bytes=8 << 30,
                   startup_trace_events=64, startup_stack_after_seconds=180, startup_stack_bytes=1 << 20,
                   application=[119, 1160], fabric=[762, 1172], offset=[4, 1],
                   mathematical_audit_during_allocation=False, automatic_retry=False, full_model=False,
                   weight_upload=dict(channels=16, maximum_tasks=4, maximum_retained_bytes=64 << 20,
                                      maximum_copy_bytes=16 << 20, batch_seconds=30),
                   weight_upload_batches=len(groups), nested_source_bytes=nested_source_bytes)
    audit = dict(seconds=900, cpu_seconds=850, address_space=2 << 30, rss=1 << 30,
                 guard_RSS_bytes=256 << 20, directory_growth_bytes=32 << 20, host_available_memory_reserve_bytes=8 << 30,
                 file_bytes=32 << 20, log_bytes=1 << 20, cpu=[0], swap=0,
                 SDK=False, model_forward=False, automatic_retry=False)
    return dict(runtime=runtime, audit=audit)


def proposal(host, profiles_by_mode, source_bytes_by_mode):
    groups = schedule([i['prepared'] for i in host['copies']['uploads'] if i['symbol'] == 'weights'])
    derived = {}
    for mode, selected in profiles_by_mode.items():
        runtime, audit = selected['runtime'], selected['audit']
        budget = host['baseline_budget' if mode == 'baseline' else 'restored_budget']
        # Nonweight copies have at most4 events. Async weights have their exact
        # own4-per-ROI +2-per-batch +1-completion ledger.256 covers fixed events,
        # both launch boundaries, startup tracing and phase/fence metadata.
        event_bound = 4*(budget['max_copies']-822)+4*822+2*len(groups)+1+256
        assert event_bound <= runtime['max_events']
        # Charge all frozen source, full journal, raw reserve, both live JSON
        # capture metadata paths, every log and total separately bounded audit.
        categories = dict(raw_reserved=runtime['max_raw_bytes'], journal_reserved=runtime['max_journal_bytes'],
                          complete_frozen_source=source_bytes_by_mode[mode], capture_and_failure_JSON=32 << 20,
                          logs_and_small_lifecycle_metadata=16 << 20, offline_output_growth=audit['directory_growth_bytes'])
        total = sum(categories.values())
        assert total < runtime['capture_directory_bytes'], (mode, total)
        derived[mode] = dict(budget=budget, journal_events_upper_bound=event_bound,
                             storage_categories=categories, conservative_storage_bytes=total,
                             directory_ceiling=runtime['capture_directory_bytes'],
                             storage_margin_bytes=runtime['capture_directory_bytes']-total)
    return dict(status='complete_source_proposal_requires_one_integrated_controller_decision',
                modes=derived, sequential_order=['baseline_physical', 'actual_release', 'baseline_offline_operator_audit',
                    'fresh_restore_admission_with_actual_checkpoint_pin', 'restore_physical', 'actual_release',
                    'restore_offline_journal_control_semantic_comparison'],
                maximum_physical_contexts=2, no_simultaneous_physical_owners=True,
                nominal_physical_execution_cap_seconds=3000, each_offline_wall_seconds=900,
                maximum_serial_execution_and_audit_windows_seconds=4800,
                cleanup=dict(inherited_module='hw00.watchdog', individual_csctl_timeout_seconds=25,
                    per_owned_job_release_poll_seconds=90, poll_sleep_seconds=2,
                    process_TERM_wait_seconds=10, process_KILL_wait_seconds=10,
                    one_expected_runtime_job=True, failure_cleanup_reserve_seconds_per_context=300,
                    scope='Release polling can cross its90s boundary by the final two25s queries.300s is the accounting reserve for one correlated job, process reap and final queries, not an extra compute or device admission.'),
                physical_windows_plus_two_cleanup_reserves_seconds=3600,
                original_prepared_array_files=650, original_prepared_array_bytes=3081379968,
                prepared_arrays_duplicated=False, new_original_shard_reads=0,
                async_weight_batches=len(groups), maximum_actual_scheduled_retained_bytes=max(g['retained_host_bytes'] for g in groups),
                host_memory=dict(client_RSS_ceiling=1 << 30, guard_RSS_ceiling=256 << 20,
                    maximum_prepared_cache=16 << 20, retained_async=64 << 20,
                    blocking_copy_working_reserve=5*(16 << 20), source_and_binding_and_raw_receipt_metadata_reserve=256 << 20,
                    source_measurements_are_not_peak_proof=True, available_RAM_reserve=8 << 30, swap_bytes=0),
                timing_basis=dict(previous_three_Linear_plus_one_Attention_capture_seconds=3*68.524565+110.465794,
                    explanation='Sum of accepted standalone host capture windows is316.039489s for almost the same calls; this is only a sizing reference. Full chain runtime and source/fsync costs remain unmeasured.',
                    original600s_capture_deadline_retained=True),
                worker=dict(SDK_request_bytes=8 << 30, SDK_CPU_millicores=1000, measured_peak=None,
                            compiler90GiB_template_not_a_runtime_worker_measurement=True),
                full_model=False, performance_claim=False, automatic_retry=False)
