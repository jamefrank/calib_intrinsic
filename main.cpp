#include "cmdline.h"
#include <spdlog/spdlog.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <opencv2/opencv.hpp>

#include <boost/filesystem.hpp>

#include "camodocal/calib/CameraCalibration.h"
#include "camodocal/chessboard/Chessboard.h"
#include "camodocal/gpl/gpl.h"

#include <iostream>
using namespace std;



int main(int argc, char *argv[])
{
  spdlog::info("Welcome to use calibCamIntrinsic calib tool!");

  cmdline::parser parser;
  parser.add<string>("input", 'i', "Input directory containing chessboard images", true, "");
  parser.add<string>("prefix", 'p', "Prefix of image name", false, "");
  parser.add<string>("extension", 'e', "File extension of images", false, ".jpg");

  parser.add<int>("board-width", 'w', "Number of inner corners on the chessboard pattern in x direction", false, 3);
  parser.add<int>("board-height", 'h', "Number of inner corners on the chessboard pattern in y direction", false, 2);
  parser.add<double>("square-size", 's', "Size of one square in mm", false, 60.0);
  parser.add<double>("marker-size", 'm', "Size of one square in mm", false, 45.0);

  parser.add<string>("camera-model", 't', "Camera model", false, "pinhole", cmdline::oneof<string>("pinhole", "fisheye", "mei"));
  parser.add<string>("camera-name", 'n', "Camera name", false, "camera");
  parser.add("charuco", '\0', "use charuco marker when calib");
  parser.add("verbose", '\0', "verbose when calib");

  parser.parse_check(argc, argv);

  // check input folder
  std::string inputDir = parser.get<std::string>("input");
  if (!boost::filesystem::exists(inputDir) && !boost::filesystem::is_directory(inputDir)) {
    spdlog::error("Cannot find input directory{}", inputDir);
    return 1;
  }

  //
  std::string cameraModel = parser.get<std::string>("camera-model");
  camodocal::Camera::ModelType modelType;
  if (boost::iequals(cameraModel, "fisheye")) {
      modelType = camodocal::Camera::KANNALA_BRANDT;
  } else if (boost::iequals(cameraModel, "mei")) {
      modelType = camodocal::Camera::MEI;
  } else if (boost::iequals(cameraModel, "pinhole")) {
      modelType = camodocal::Camera::PINHOLE;
  } else if (boost::iequals(cameraModel, "pinhole_full")) {
      modelType = camodocal::Camera::PINHOLE_FULL;
  } else if (boost::iequals(cameraModel, "scaramuzza")) {
      modelType = camodocal::Camera::SCARAMUZZA;
  } else {
    spdlog::error("# ERROR: Unknown camera model: {}", cameraModel);
    return 1;
  }

  // look for images in input directory
  std::vector<std::string> imageFilenames;
  boost::filesystem::directory_iterator itr;
  std::string prefix = parser.get<std::string>("prefix");
  std::string fileExtension = parser.get<std::string>("extension");
  for (boost::filesystem::directory_iterator itr(inputDir); itr != boost::filesystem::directory_iterator(); ++itr) {
      if (!boost::filesystem::is_regular_file(itr->status())) {
          continue;
      }
      std::string filename = itr->path().filename().string();
      // check if prefix matches
      if (!prefix.empty()) {
          if (filename.compare(0, prefix.length(), prefix) != 0) {
              continue;
          }
      }
      // check if file extension matches
      if (filename.compare(filename.length() - fileExtension.length(), fileExtension.length(), fileExtension) != 0) {
          continue;
      }

      imageFilenames.push_back(itr->path().string());
  }

  if (imageFilenames.empty()) {
      spdlog::error("# ERROR: No chessboard images found.");
      return 1;
  }
  else
    spdlog::info("# INFO: # images: {}", imageFilenames.size());
  std::sort(imageFilenames.begin(), imageFilenames.end());

  // detect markers
  spdlog::info("# INFO: # start detect markers ...");
  bool verbose = parser.exist("verbose");
  std::string cameraName = parser.get<std::string>("camera-name");
  cv::Size boardSize;
  boardSize.width = parser.get<int>("board-width");
  boardSize.height = parser.get<int>("board-height");
  float squareSize = parser.get<double>("square-size");
  float markerSize = parser.get<double>("marker-size");
  bool useOpenCV = true;
  bool useCharuco = parser.exist("charuco");
  
  cv::Mat image = cv::imread(imageFilenames.front(), -1);
  const cv::Size frameSize = image.size();
  const int MAX_SCREEN_WIDTH = 960;  // 假设最大屏幕宽度
  const int MAX_SCREEN_HEIGHT = 540;
  double scale_x = (double)MAX_SCREEN_WIDTH / frameSize.width;
  double scale_y = (double)MAX_SCREEN_HEIGHT / frameSize.height;
  double scale = std::min({1.0, scale_x, scale_y});

  camodocal::CameraCalibration calibration(modelType, cameraName, frameSize, boardSize, squareSize);
  calibration.setVerbose(verbose);

  std::vector<bool> chessboardFound(imageFilenames.size(), false);
  for (size_t i = 0; i < imageFilenames.size(); ++i) {
      image = cv::imread(imageFilenames.at(i), -1);

      camodocal::Chessboard chessboard(boardSize, image, squareSize, markerSize);

      chessboard.findCorners(useOpenCV, useCharuco);
      if (chessboard.cornersFound()) {
          if (verbose) {
              spdlog::info("# INFO: Detected chessboard in image {},{}", i + 1, imageFilenames.at(i));
          }

          calibration.addChessboardData(chessboard.getCorners());

          cv::Mat sketch;
          chessboard.getSketch().copyTo(sketch);
          cv::Mat resizedImg;

          if (scale < 1.0) {
              cv::resize(sketch, resizedImg, cv::Size(), scale, scale, cv::INTER_AREA);
              cv::imshow("Image", resizedImg);
          } else {
              cv::imshow("Image", sketch);
          }
          cv::waitKey(50);
      } else if (verbose) {
        spdlog::warn("# INFO: Did not detect chessboard in image {}", i + 1);
      }
      chessboardFound.at(i) = chessboard.cornersFound();
  }

  if (calibration.sampleCount() < 10) {
      spdlog::error("# ERROR: Insufficient number of detected chessboards {} < 10.", calibration.sampleCount());
      return 1;
  }

  // start calib
  spdlog::info("Calibrating start ...");

  double startTime = camodocal::timeInSeconds();

  calibration.calibrate();
  calibration.writeParams(cameraName + "_camera_calib.yaml");
  calibration.writeChessboardData(cameraName + "_chessboard_data.dat");

  double elapsed = camodocal::timeInSeconds() - startTime;
  spdlog::info("# INFO: Calibration took a total time of {:.3f} sec.", elapsed);
  spdlog::info("# INFO: Wrote calibration file to {}_camera_calib.yaml", cameraName);
 

  if (verbose) {
      std::vector<cv::Mat> cbImages;
      std::vector<std::string> cbImageFilenames;

      for (size_t i = 0; i < imageFilenames.size(); ++i) {
          if (!chessboardFound.at(i)) {
              continue;
          }

          cbImages.push_back(cv::imread(imageFilenames.at(i), -1));
          cbImageFilenames.push_back(imageFilenames.at(i));
      }

      // visualize observed and reprojected points
      calibration.drawResults(cbImages);



      for (size_t i = 0; i < cbImages.size(); ++i) {
        cv::putText(cbImages.at(i), cbImageFilenames.at(i), cv::Point(10, 20),
                    cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

        cv::Mat resizedImg;
        if (scale < 1.0) {
            cv::resize(cbImages[i], resizedImg, cv::Size(), scale, scale, cv::INTER_AREA);
            cv::imshow("Image", resizedImg);
        } else {
            cv::imshow("Image", cbImages[i]);
        }
        cv::waitKey(0);
    }
    cv::destroyWindow("Image");
  }
  
  return 0;
}