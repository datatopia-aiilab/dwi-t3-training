"""
Script to check MLflow runs and artifacts
"""

import mlflow
from pathlib import Path
import config

def list_recent_runs(n=5):
    """List recent MLflow runs"""
    
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    
    # Get experiment
    experiment = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
    
    if not experiment:
        print(f"❌ Experiment '{config.MLFLOW_EXPERIMENT_NAME}' not found")
        return
    
    print(f"\n{'='*80}")
    print(f"📊 MLflow Experiment: {config.MLFLOW_EXPERIMENT_NAME}")
    print(f"   Experiment ID: {experiment.experiment_id}")
    print(f"   Tracking URI: {config.MLFLOW_TRACKING_URI}")
    print(f"{'='*80}\n")
    
    # Get runs
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=n
    )
    
    if runs.empty:
        print("❌ No runs found")
        return
    
    print(f"🏃 Recent Runs (showing {len(runs)} most recent):\n")
    
    for idx, run in runs.iterrows():
        run_id = run['run_id']
        status = run['status']
        start_time = run['start_time']
        
        # Get metrics
        best_dice = run.get('metrics.best_val_dice', 'N/A')
        test_dice = run.get('metrics.test_dice', 'N/A')
        
        print(f"   Run {idx + 1}:")
        print(f"      ID: {run_id}")
        print(f"      Status: {status}")
        print(f"      Start: {start_time}")
        
        if best_dice != 'N/A':
            print(f"      Best Val Dice: {best_dice:.4f}")
        
        if test_dice != 'N/A':
            print(f"      Test Dice: {test_dice:.4f}")
        
        # Check artifacts
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        
        if artifacts:
            print(f"      Artifacts:")
            for artifact in artifacts:
                if artifact.is_dir:
                    # Count files in directory
                    sub_artifacts = client.list_artifacts(run_id, artifact.path)
                    count = len(sub_artifacts)
                    print(f"         📁 {artifact.path}/ ({count} files)")
                else:
                    print(f"         📄 {artifact.path}")
        else:
            print(f"      ❌ No artifacts logged")
        
        print()
    
    # Check for specific artifacts in latest run
    if len(runs) > 0:
        latest_run_id = runs.iloc[0]['run_id']
        print(f"\n{'='*80}")
        print(f"🔍 Detailed Check of Latest Run ({latest_run_id}):")
        print(f"{'='*80}\n")
        
        client = mlflow.tracking.MlflowClient()
        
        # Check for specific files
        expected_artifacts = [
            "plots/training_curves_advanced.png",
            "plots/test_metrics_distribution.png",
            "evaluation/test_per_sample_results.csv",
            "predictions/",
            "models/best_model/",
        ]
        
        print("Expected Artifacts:")
        for artifact_path in expected_artifacts:
            try:
                if artifact_path.endswith('/'):
                    # Check directory
                    artifacts = client.list_artifacts(latest_run_id, artifact_path.rstrip('/'))
                    if artifacts:
                        print(f"   ✅ {artifact_path} ({len(artifacts)} items)")
                    else:
                        print(f"   ❌ {artifact_path} (empty or not found)")
                else:
                    # Check file
                    artifacts = client.list_artifacts(latest_run_id, Path(artifact_path).parent.as_posix())
                    file_found = any(a.path == artifact_path for a in artifacts)
                    if file_found:
                        print(f"   ✅ {artifact_path}")
                    else:
                        print(f"   ❌ {artifact_path} (not found)")
            except Exception as e:
                print(f"   ⚠️  {artifact_path} (error checking: {e})")
        
        # Show metrics
        run_data = client.get_run(latest_run_id)
        metrics = run_data.data.metrics
        
        print(f"\n{'='*80}")
        print(f"📊 Latest Run Metrics:")
        print(f"{'='*80}\n")
        
        # Group metrics
        training_metrics = {k: v for k, v in metrics.items() if 'train' in k or 'val' in k or 'best' in k}
        test_metrics = {k: v for k, v in metrics.items() if 'test' in k}
        
        if training_metrics:
            print("Training Metrics:")
            for k, v in sorted(training_metrics.items()):
                print(f"   {k}: {v:.4f}")
        
        if test_metrics:
            print(f"\nTest Metrics:")
            for k, v in sorted(test_metrics.items()):
                print(f"   {k}: {v:.4f}")
            
            # Check for volume metrics
            volume_metrics = [k for k in test_metrics.keys() if 'volume' in k]
            if volume_metrics:
                print(f"\n   ✅ Volume metrics found: {len(volume_metrics)}")
                for vm in volume_metrics:
                    print(f"      - {vm}: {test_metrics[vm]:.4f}")
            else:
                print(f"\n   ❌ No volume metrics found")
        
        print(f"\n{'='*80}")
        print(f"💡 To view in UI:")
        print(f"   mlflow ui --port 5000")
        print(f"   Then open: http://localhost:5000")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    list_recent_runs(n=5)
