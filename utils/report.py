import os
import time
from datetime import datetime

class ReportGenerator:
    def __init__(self, start_time, end_time, prompt, input_image=None, output_image=None):
        """
        Initializes the ReportGenerator.
        
        Args:
            start_time (float): System time (time.time()) when execution started.
            end_time (float): System time (time.time()) when execution finished.
            prompt (str): Prompt used for generation or visual query.
            input_image (str, optional): Path to the input image (if applicable).
            output_image (str, optional): Path to the generated/output image.
        """
        self.start_time = start_time
        self.end_time = end_time
        self.prompt = prompt
        self.input_image = input_image
        self.output_image = output_image
        self.execution_time = end_time - start_time
        self.generate_report()

    def generate_report(self):
        # Resolve the project root dynamically
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        reports_dir = os.path.join(project_root, "outputs", "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"report_{timestamp}.txt"
        report_path = os.path.join(reports_dir, report_filename)
        
        # Build relative paths for the report file so it is clean
        input_rel = os.path.relpath(self.input_image, project_root) if self.input_image else "N/A"
        output_rel = os.path.relpath(self.output_image, project_root) if self.output_image else "N/A"
        
        report_content = f"""========================================
MODEL EXECUTION REPORT
========================================
Date/Time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Start Time:      {time.ctime(self.start_time)}
End Time:        {time.ctime(self.end_time)}
Total Duration:  {self.execution_time:.4f} seconds
----------------------------------------
Prompt:          {self.prompt}
Input Image:     {input_rel}
Output Image:    {output_rel}
========================================
"""
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        print(f"Execution report successfully written to: {report_path}")
