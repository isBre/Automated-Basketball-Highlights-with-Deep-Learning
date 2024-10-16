def read_from_text(text_path: str) -> str:
  """
  Reads a file from text.
  The files will usually consist of a single line containing
  the minutes of the baskets (grand truth)
  Args:
        text_path: path of the txt file

  Returns:
        points_text: content inside the txt file in path text_path
  """
  points_text = ""
  with open(text_path, 'r') as f:
      for line in f:
          points_text += line
  return points_text

def hms_to_s(hms: str) -> int:
  """
  Transforms an hour, minute and second format string into seconds 
  Args:
        hms: str in format hour, minute and second
  Returns:
        seconds: seconds corresponding to the format hours minutes seconds
  """
  if len(hms.split(':')) == 1:
    return int(hms)
  elif len(hms.split(':')) == 2:
    m = int(hms.split(':')[0])
    s = int(hms.split(':')[1])
    return m * 60 + s
  else:
    h = int(hms.split(':')[0])
    m = int(hms.split(':')[1])
    s = int(hms.split(':')[2])
    return h * 3600 + m * 60 + s
  
def frame_to_hms(frame: int, fps: int) -> str:
  """
  Transforms a frame count into hour, minute and second format string 
  Args:
        frame: frame count of a particular video
        fps: frame per seconds of a particular video
  Returns:
        hms: str in hour, minute and second format
  """
  all_sec = frame // fps
  all_m = all_sec // 60
  s = all_sec % 60
  m = all_m % 60
  h = all_m // 60
  return f"{h}:{m}:{s}"