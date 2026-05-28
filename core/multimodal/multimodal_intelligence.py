"""Multimodal Intelligence for chart detection, table summarization, OCR, and image captioning."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
import logging
from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)


@dataclass
class ChartInfo:
    """Information about detected chart."""
    chart_type: str  # bar, line, pie, scatter, etc.
    title: str
    x_axis: str
    y_axis: str
    data_points: List[Dict[str, Any]]
    confidence: float


@dataclass
class TableInfo:
    """Information about detected table."""
    headers: List[str]
    rows: List[List[str]]
    summary: str
    confidence: float


class ChartDetector:
    """Detects and extracts information from charts with OCR enhancement."""
    
    def __init__(self):
        """Initialize the chart detector."""
        self.chart_patterns = {
            "bar": [r"bar chart", r"bar graph", r"histogram", r"column chart"],
            "line": [r"line chart", r"line graph", r"trend", r"over time", r"time series"],
            "pie": [r"pie chart", r"pie graph", r"distribution", r"percentage"],
            "scatter": [r"scatter plot", r"scatter graph", r"correlation"],
            "area": [r"area chart", r"area graph"],
            "radar": [r"radar chart", r"spider chart"]
        }
        
        logger.info("ChartDetector initialized")
    
    def detect(self, image: Image.Image, caption: str = "") -> Optional[ChartInfo]:
        """Detect chart type and extract information with OCR enhancement.
        
        Args:
            image: PIL Image
            caption: Optional caption text
            
        Returns:
            ChartInfo if chart detected, None otherwise
        """
        logger.info("Detecting chart with OCR enhancement")
        
        # Extract text from image using OCR
        ocr_text = self._extract_ocr_text(image)
        
        # Combine OCR text with caption
        combined_text = f"{caption} {ocr_text}".lower()
        
        # Detect chart type
        chart_type = self._infer_chart_type(combined_text)
        if chart_type:
            return self._extract_chart_info(chart_type, combined_text, ocr_text)
        
        logger.info("No chart detected")
        return None
    
    def _extract_ocr_text(self, image: Image.Image) -> str:
        """Extract text from image using OCR.
        
        Args:
            image: PIL Image
            
        Returns:
            Extracted text
        """
        try:
            text = pytesseract.image_to_string(image)
            logger.info(f"Extracted {len(text)} characters via OCR for chart detection")
            return text.strip()
        except Exception as e:
            logger.error(f"OCR extraction failed for chart: {e}")
            return ""
    
    def _infer_chart_type(self, text: str) -> Optional[str]:
        """Infer chart type from text.
        
        Args:
            text: Text description
            
        Returns:
            Chart type if inferred, None otherwise
        """
        text_lower = text.lower()
        
        for chart_type, patterns in self.chart_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return chart_type
        
        return None
    
    def _extract_chart_info(self, chart_type: str, combined_text: str, ocr_text: str) -> ChartInfo:
        """Extract chart information from combined text and OCR.
        
        Args:
            chart_type: Detected chart type
            combined_text: Combined caption and OCR text
            ocr_text: OCR extracted text
            
        Returns:
            ChartInfo object
        """
        # Extract title (first sentence or from OCR)
        title = self._extract_title(combined_text)
        
        # Extract axes (enhanced with OCR data)
        x_axis, y_axis = self._extract_axes(combined_text)
        
        # Extract data points from OCR
        data_points = self._extract_data_points_from_ocr(ocr_text)
        
        return ChartInfo(
            chart_type=chart_type,
            title=title,
            x_axis=x_axis,
            y_axis=y_axis,
            data_points=data_points,
            confidence=0.8
        )
    
    def _extract_title(self, text: str) -> str:
        """Extract chart title from text.
        
        Args:
            text: Text content
            
        Returns:
            Extracted title
        """
        # Look for common title patterns
        title_patterns = [
            r"title:\s*(.+)",
            r"chart:\s*(.+)",
            r"graph:\s*(.+)"
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback to first sentence
        sentences = text.split('.')
        if sentences:
            return sentences[0].strip()[:100]  # Limit to 100 chars
        
        return "Unknown Chart"
    
    def _extract_axes(self, text: str) -> Tuple[str, str]:
        """Extract axis labels from text.
        
        Args:
            text: Text content
            
        Returns:
            Tuple of (x_axis, y_axis)
        """
        x_axis = "Unknown"
        y_axis = "Unknown"
        
        # Look for axis patterns
        if "x-axis" in text.lower() or "x axis" in text.lower():
            match = re.search(r"x[- ]axis[:\s]+(.+)", text, re.IGNORECASE)
            if match:
                x_axis = match.group(1).strip()
        
        if "y-axis" in text.lower() or "y axis" in text.lower():
            match = re.search(r"y[- ]axis[:\s]+(.+)", text, re.IGNORECASE)
            if match:
                y_axis = match.group(1).strip()
        
        # Fallback to common axis keywords
        axis_keywords = {
            "x": ["time", "year", "date", "month", "category"],
            "y": ["value", "amount", "count", "percentage", "salary", "package"]
        }
        
        if x_axis == "Unknown":
            for keyword in axis_keywords["x"]:
                if keyword in text.lower():
                    x_axis = keyword.title()
                    break
        
        if y_axis == "Unknown":
            for keyword in axis_keywords["y"]:
                if keyword in text.lower():
                    y_axis = keyword.title()
                    break
        
        return x_axis, y_axis
    
    def _extract_data_points_from_ocr(self, ocr_text: str) -> List[Dict[str, Any]]:
        """Extract data points from OCR text.
        
        Args:
            ocr_text: OCR extracted text
            
        Returns:
            List of data point dictionaries
        """
        data_points = []
        
        # Look for numeric patterns (values and labels)
        lines = ocr_text.split('\n')
        for line in lines:
            # Look for lines with numbers
            numbers = re.findall(r'\d+\.?\d*', line)
            if len(numbers) >= 1:
                # Extract label (non-numeric part)
                label = re.sub(r'\d+\.?\d*', '', line).strip()
                if label:
                    data_points.append({
                        "label": label,
                        "value": numbers[0] if len(numbers) == 1 else numbers
                    })
        
        return data_points[:10]  # Limit to 10 data points


class TableSummarizer:
    """Summarizes table data for better retrieval."""
    
    def __init__(self):
        """Initialize the table summarizer."""
        logger.info("TableSummarizer initialized")
    
    def summarize(self, table_data: List[List[str]], headers: List[str]) -> TableInfo:
        """Summarize table data.
        
        Args:
            table_data: Table rows
            headers: Table headers
            
        Returns:
            TableInfo object
        """
        logger.info(f"Summarizing table with {len(table_data)} rows")
        
        # Lazy import pandas to avoid circular import
        import pandas as pd
        
        # Create DataFrame for analysis
        df = pd.DataFrame(table_data, columns=headers)
        
        # Generate summary
        summary = self._generate_summary(df)
        
        return TableInfo(
            headers=headers,
            rows=table_data,
            summary=summary,
            confidence=0.8
        )
    
    def _generate_summary(self, df) -> str:
        """Generate natural language summary of table.
        
        Args:
            df: DataFrame
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        # Basic statistics
        summary_parts.append(f"Table contains {len(df)} rows and {len(df.columns)} columns")
        
        # Column information
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_values = df[col].nunique()
                summary_parts.append(f"Column '{col}' has {unique_values} unique values")
            else:
                mean_val = df[col].mean()
                summary_parts.append(f"Column '{col}' has mean {mean_val:.2f}")
        
        return ". ".join(summary_parts)


class OCRProcessor:
    """OCR processing for scanned PDFs and images."""
    
    def __init__(self):
        """Initialize the OCR processor."""
        try:
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            self.available = True
            logger.info("OCRProcessor initialized")
        except Exception as e:
            self.available = False
            logger.warning(f"OCR not available: {e}")
    
    def extract_text(self, image: Image.Image) -> str:
        """Extract text from image using OCR.
        
        Args:
            image: PIL Image
            
        Returns:
            Extracted text
        """
        if not self.available:
            logger.warning("OCR not available")
            return ""
        
        try:
            text = pytesseract.image_to_string(image)
            logger.info(f"Extracted {len(text)} characters via OCR")
            return text
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""
    
    def extract_with_layout(self, image: Image.Image) -> Dict[str, Any]:
        """Extract text with layout information.
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with text and layout information
        """
        if not self.available:
            return {"text": "", "layout": {}}
        
        try:
            # Get OCR data with bounding boxes
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Extract text blocks
            text_blocks = []
            for i in range(len(data['text'])):
                if data['text'][i].strip():
                    text_blocks.append({
                        "text": data['text'][i],
                        "bbox": (
                            data['left'][i],
                            data['top'][i],
                            data['left'][i] + data['width'][i],
                            data['top'][i] + data['height'][i]
                        ),
                        "confidence": data['conf'][i]
                    })
            
            full_text = " ".join([block['text'] for block in text_blocks])
            
            logger.info(f"Extracted {len(text_blocks)} text blocks via OCR")
            
            return {
                "text": full_text,
                "layout": {
                    "text_blocks": text_blocks,
                    "image_size": image.size
                }
            }
        except Exception as e:
            logger.error(f"OCR with layout failed: {e}")
            return {"text": "", "layout": {}}


class ImageCaptioner:
    """Generates captions for images."""
    
    def __init__(self):
        """Initialize the image captioner."""
        logger.info("ImageCaptioner initialized")
    
    def caption(self, image: Image.Image) -> str:
        """Generate caption for image using OCR text extraction.
        
        Args:
            image: PIL Image
            
        Returns:
            Generated caption based on OCR text
        """
        # Use OCR to extract text as caption
        try:
            text = pytesseract.image_to_string(image)
            if text.strip():
                return f"Image contains text: {text.strip()[:100]}"
            return "Image contains no readable text"
        except Exception as e:
            logger.error(f"Image captioning failed: {e}")
            return "Unable to extract image content"
    
    def caption_with_objects(self, image: Image.Image) -> Dict[str, Any]:
        """Generate caption with detected objects using OCR.
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with caption and detected text objects
        """
        try:
            text = pytesseract.image_to_string(image)
            words = text.strip().split() if text.strip() else []
            return {
                "caption": f"Image contains {len(words)} words" if words else "Image contains no readable text",
                "objects": words[:10]  # Return first 10 words as detected objects
            }
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return {
                "caption": "Unable to extract image content",
                "objects": []
            }


class MultimodalIntelligence:
    """Unified multimodal intelligence system."""
    
    def __init__(self):
        """Initialize the multimodal intelligence system."""
        self.chart_detector = ChartDetector()
        self.table_summarizer = TableSummarizer()
        self.ocr_processor = OCRProcessor()
        self.image_captioner = ImageCaptioner()
        
        logger.info("MultimodalIntelligence initialized")
    
    def process_image(self, image: Image.Image, context: str = "") -> Dict[str, Any]:
        """Process image with all multimodal capabilities.
        
        Args:
            image: PIL Image
            context: Optional context text
            
        Returns:
            Dictionary with all extracted information
        """
        logger.info("Processing image with multimodal intelligence")
        
        results = {
            "chart_info": None,
            "table_info": None,
            "ocr_text": "",
            "caption": "",
            "metadata": {}
        }
        
        # Try to detect chart
        chart_info = self.chart_detector.detect(image, context)
        if chart_info:
            results["chart_info"] = {
                "chart_type": chart_info.chart_type,
                "title": chart_info.title,
                "x_axis": chart_info.x_axis,
                "y_axis": chart_info.y_axis,
                "confidence": chart_info.confidence
            }
        
        # Extract text via OCR
        ocr_text = self.ocr_processor.extract_text(image)
        if ocr_text:
            results["ocr_text"] = ocr_text
            results["metadata"]["ocr_available"] = True
        
        # Generate caption
        caption = self.image_captioner.caption(image)
        results["caption"] = caption
        
        logger.info("Image processing complete")
        return results
    
    def process_table(self, table_data: List[List[str]], headers: List[str]) -> Dict[str, Any]:
        """Process table data.
        
        Args:
            table_data: Table rows
            headers: Table headers
            
        Returns:
            Dictionary with table information
        """
        logger.info("Processing table")
        
        table_info = self.table_summarizer.summarize(table_data, headers)
        
        return {
            "headers": table_info.headers,
            "row_count": len(table_info.rows),
            "summary": table_info.summary,
            "confidence": table_info.confidence
        }
    
    def extract_visual_evidence(
        self,
        image: Image.Image,
        query: str
    ) -> Optional[Dict[str, Any]]:
        """Extract visual evidence relevant to query.
        
        Args:
            image: PIL Image
            query: User query
            
        Returns:
            Visual evidence if found, None otherwise
        """
        logger.info(f"Extracting visual evidence for query: {query}")
        
        # Process image
        image_info = self.process_image(image)
        
        # Check if image info is relevant to query
        relevance_score = self._calculate_relevance(image_info, query)
        
        if relevance_score > 0.5:
            return {
                "evidence": image_info,
                "relevance_score": relevance_score,
                "source": "visual"
            }
        
        return None
    
    def _calculate_relevance(self, image_info: Dict[str, Any], query: str) -> float:
        """Calculate relevance of image to query.
        
        Args:
            image_info: Image information
            query: User query
            
        Returns:
            Relevance score between 0 and 1
        """
        query_lower = query.lower()
        score = 0.0
        
        # Check OCR text relevance
        if image_info.get("ocr_text"):
            ocr_text = image_info["ocr_text"].lower()
            if query_lower in ocr_text:
                score += 0.5
            elif any(word in ocr_text for word in query_lower.split()):
                score += 0.3
        
        # Check caption relevance
        if image_info.get("caption"):
            caption = image_info["caption"].lower()
            if query_lower in caption:
                score += 0.3
        
        # Check chart info relevance
        if image_info.get("chart_info"):
            chart_info = image_info["chart_info"]
            if any(word in str(chart_info).lower() for word in query_lower.split()):
                score += 0.2
        
        return min(score, 1.0)
