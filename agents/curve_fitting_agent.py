from agents.base import BaseAgent
from tools.fitting_agent import (
    LLMClient, CurveFittingConfig, build_agent_config,
    curate_dataset, run_complete_analysis, get_xy_for_well,
    save_all_wells_results, export_peak_data_to_csv
)
import streamlit as st
from tools.memory import MemoryManager
import os
import tempfile
import json
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def _display_fitting_plot(result: Dict[str, Any], container, well_name: str):
    """Display fitting plot in real-time during analysis."""
    try:
        # Try to load from saved plot file first
        if 'files' in result and 'fitting_plot' in result['files']:
            plot_path = result['files']['fitting_plot']
            if os.path.exists(plot_path):
                with container:
                    read_info = f" Read {result.get('read', '')}" if result.get('read') else ""
                    st.subheader(f"📊 Well {well_name}{read_info} - R²: {result['fit_result'].stats.r2:.4f}")
                    img = mpimg.imread(plot_path)
                    st.image(img, caption=f"Fitting plot for well {well_name}{read_info}", use_container_width=True)
                    return
        
        # If no saved plot, create one on-the-fly from data
        if 'data' in result and 'fit_result' in result:
            x = result['data']['x']
            y = result['data']['y']
            fit_result = result['fit_result']
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Top plot: data and fit
            ax1.plot(x, y, 'b-', label='Data', linewidth=1.5, alpha=0.7)
            if fit_result.lmfit_result is not None:
                ax1.plot(x, fit_result.lmfit_result.best_fit, 'r-', label='Fit', linewidth=2)
            
            # Plot individual peaks if available
            if fit_result.peaks:
                for i, peak in enumerate(fit_result.peaks):
                    if hasattr(peak, 'center') and fit_result.lmfit_result is not None:
                        # Create peak curve
                        peak_curve = _create_peak_curve(x, peak, fit_result.lmfit_result)
                        if peak_curve is not None:
                            ax1.plot(x, peak_curve, '--', alpha=0.6, label=f'Peak {i+1}')
            
            ax1.set_xlabel('Wavelength (nm)')
            ax1.set_ylabel('Intensity')
            ax1.set_title(f'Peak Fitting - Well {well_name} (R² = {fit_result.stats.r2:.4f})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Bottom plot: residuals
            if fit_result.lmfit_result is not None:
                residuals = y - fit_result.lmfit_result.best_fit
                ax2.plot(x, residuals, 'g-', linewidth=1)
                ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                ax2.set_xlabel('Wavelength (nm)')
                ax2.set_ylabel('Residuals')
                ax2.set_title('Residuals')
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Display in Streamlit
            with container:
                read_info = f" Read {result.get('read', '')}" if result.get('read') else ""
                st.subheader(f"📊 Well {well_name}{read_info} - R²: {fit_result.stats.r2:.4f}")
                st.pyplot(fig)
                plt.close(fig)
    except Exception as e:
        # If plotting fails, just show a message
        with container:
            st.warning(f"Could not display plot for well {well_name}: {str(e)}")


def _create_peak_curve(x: np.ndarray, peak, lmfit_result) -> Optional[np.ndarray]:
    """Create a curve for a single peak component."""
    try:
        if not hasattr(peak, 'center') or lmfit_result is None:
            return None
        
        # Try to extract peak parameters from lmfit result
        # This is a simplified version - you may need to adjust based on your peak model
        center = peak.center
        amplitude = getattr(peak, 'amplitude', getattr(peak, 'height', 1.0))
        sigma = getattr(peak, 'sigma', getattr(peak, 'fwhm', 10.0) / 2.3548)
        
        # Create Gaussian peak
        peak_curve = amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)
        return peak_curve
    except:
        return None


class CurveFittingAgent(BaseAgent):
    """Curve fitting agent using Spectropus-style LLM-assisted multi-peak fitting."""

    def __init__(self, name: str = "Curve Fitting Agent", desc: str = "LLM-assisted multi-peak curve fitting"):
        super().__init__(name, desc)
        self.memory = MemoryManager()
        self.llm_client = None

    def _get_llm_client(self, min_delay_seconds: Optional[float] = None) -> LLMClient:
        """Initialize or return cached LLM client."""
        # Get delay from provided value or session state or default
        delay = min_delay_seconds or st.session_state.get('gemini_delay_seconds', 0.5)
        
        # Check if we need to recreate the client (if delay changed or client doesn't exist)
        if self.llm_client is None or (hasattr(self.llm_client, 'min_delay') and self.llm_client.min_delay != delay):
            if not st.session_state.get('api_key'):
                raise ValueError("API key not found. Please set your API key in Settings.")
            
            self.llm_client = LLMClient(
                provider="gemini",
                model_id="gemini-2.5-flash-preview-image",  # Image-capable model (fallback to lite for text-only)
                api_key=st.session_state.api_key,
                min_delay_seconds=delay
            )
        return self.llm_client

    def confidence(self, payload: Dict[str, Any]) -> float:
        """Return confidence score for curve fitting tasks."""
        # Curve fitting is generally applicable to spectral data
        return 0.8

    def run_agent(self, memory: MemoryManager) -> None:
        """
        Render UI and handle agent interactions.
        
        Note: The curve fitting UI is primarily handled in the curve_fitting.py page,
        but this method satisfies the abstract base class requirement.
        """
        # The UI is handled in pages/curve_fitting.py
        # This method exists to satisfy the abstract base class requirement
        pass

    def run_curve_fitting(
        self,
        data_csv_path: str,
        composition_csv_path: str,
        wells_to_analyze: Optional[List[str]] = None,
        reads_to_analyze: Optional[str] = "auto",
        max_peaks: int = 4,
        r2_target: float = 0.90,
        max_attempts: int = 3,
        save_plots: bool = True,
        start_wavelength: Optional[int] = None,
        end_wavelength: Optional[int] = None,
        wavelength_step_size: Optional[int] = None,
        api_delay_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run complete curve fitting analysis using Spectropus methodology.

        Args:
            data_csv_path: Path to the spectral data CSV
            composition_csv_path: Path to the composition CSV
            wells_to_analyze: List of wells to analyze (None = all)
            reads_to_analyze: Reads to analyze ("auto", "all", or comma-separated)
            max_peaks: Maximum number of peaks to fit
            r2_target: Target R² value for good fit
            max_attempts: Maximum fitting attempts per well
            save_plots: Whether to save fitting plots
            start_wavelength: Minimum wavelength to include (None = use full range)
            end_wavelength: Maximum wavelength to include (None = use full range)
            wavelength_step_size: Step size for wavelength sampling (None = use all data points)
            api_delay_seconds: Delay between API calls to prevent rate limiting (None = use default 0.5s)

        Returns:
            Dictionary with analysis results
        """
        try:
            # Validate inputs
            if not os.path.exists(data_csv_path):
                raise FileNotFoundError(f"Data CSV not found: {data_csv_path}")
            if not os.path.exists(composition_csv_path):
                raise FileNotFoundError(f"Composition CSV not found: {composition_csv_path}")

            # Build configuration
            config = build_agent_config(
                data_csv=data_csv_path,
                composition_csv=composition_csv_path,
                read_selection=reads_to_analyze,
                wells_to_ignore=None if wells_to_analyze is None else [],
                start_wavelength=start_wavelength,
                end_wavelength=end_wavelength,
                wavelength_step_size=wavelength_step_size,
                fill_na_value=0.0
            )

            # Get LLM client with rate limiting delay
            llm = self._get_llm_client(min_delay_seconds=api_delay_seconds)

            # Curate dataset
            curated = curate_dataset(config)
            available_wells = curated["wells"]

            # Filter wells if specified
            if wells_to_analyze:
                wells_to_process = [w for w in wells_to_analyze if w in available_wells]
            else:
                wells_to_process = available_wells

            if not wells_to_process:
                raise ValueError("No valid wells found to analyze")

            st.info(f"Starting analysis of {len(wells_to_process)} wells: {wells_to_process}")

            # Analyze each well
            all_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Create a container for real-time plots
            plot_container = st.container()
            
            for i, well_name in enumerate(wells_to_process):
                status_text.text(f"Analyzing well {well_name} ({i+1}/{len(wells_to_process)})")
                progress_bar.progress((i) / len(wells_to_process))

                try:
                    # Run complete analysis for this well
                    result = run_complete_analysis(
                        config=config,
                        well_name=well_name,
                        llm=llm,
                        reads=reads_to_analyze,
                        max_peaks=max_peaks,
                        r2_target=r2_target,
                        max_attempts=max_attempts,
                        save_plots=save_plots
                    )

                    # Handle both single result and list of results (multiple reads)
                    if isinstance(result, list):
                        for r in result:
                            all_results.append(r)
                            # Display plot for each read result
                            _display_fitting_plot(r, plot_container, well_name)
                    else:
                        all_results.append(result)
                        # Display plot for single result
                        _display_fitting_plot(result, plot_container, well_name)

                except Exception as e:
                    st.error(f"Error analyzing well {well_name}: {str(e)}")
                    import traceback
                    st.exception(e)
                    continue

            progress_bar.progress(1.0)
            status_text.text("Analysis complete!")

            # Save consolidated results
            if all_results:
                # Save comprehensive JSON results
                json_filename = "results/all_wells_analysis.json"
                json_file = save_all_wells_results(all_results, json_filename)

                # Export to CSV
                csv_filename = "results/peak_data_export.csv"
                csv_file = export_peak_data_to_csv(all_results, csv_filename)

                return {
                    "success": True,
                    "results": all_results,
                    "files": {
                        "json_results": json_file,
                        "csv_export": csv_file
                    },
                    "summary": {
                        "total_wells": len(all_results),
                        "successful_fits": len([r for r in all_results if r['fit_result'].success]),
                        "wells_analyzed": wells_to_process
                    }
                }
            else:
                return {
                    "success": False,
                    "error": "No wells were successfully analyzed",
                    "results": []
                }

        except Exception as e:
            st.error(f"Curve fitting analysis failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "results": []
            }

    def analyze_single_well(
        self,
        data_csv_path: str,
        composition_csv_path: str,
        well_name: str,
        read: Optional[int] = None,
        max_peaks: int = 4,
        model_kind: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a single well with detailed output for interactive use.

        Args:
            data_csv_path: Path to spectral data CSV
            composition_csv_path: Path to composition CSV
            well_name: Name of well to analyze
            read: Specific read number (None = auto)
            max_peaks: Maximum peaks to fit
            model_kind: Peak model type (None = auto-select)

        Returns:
            Analysis results for the single well
        """
        try:
            # Build configuration
            config = build_agent_config(
                data_csv=data_csv_path,
                composition_csv=composition_csv_path,
                read_selection="all",
                wells_to_ignore=[],
                fill_na_value=0.0
            )

            # Get LLM client
            llm = self._get_llm_client()

            # Run analysis
            result = run_complete_analysis(
                config=config,
                well_name=well_name,
                llm=llm,
                reads=read or "auto",
                max_peaks=max_peaks,
                model_kind=model_kind,
                save_plots=True
            )

            return {
                "success": True,
                "result": result,
                "well_name": well_name,
                "read": read
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "well_name": well_name,
                "read": read
            }