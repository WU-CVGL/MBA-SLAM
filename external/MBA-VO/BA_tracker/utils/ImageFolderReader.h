#ifndef UTILS_IMAGE_FOLDER_READER_H
#define UTILS_IMAGE_FOLDER_READER_H

#include <algorithm>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace SLAM
{
    namespace Utils
    {
        inline bool isimage(std::string &filename)
        {
            std::vector<std::string> image_extensions;
            image_extensions.push_back(".jpg");
            image_extensions.push_back(".JPG");
            image_extensions.push_back(".png");
            image_extensions.push_back(".PNG");
            image_extensions.push_back(".jpeg");
            image_extensions.push_back(".JPEG");
            image_extensions.push_back(".tif");
            image_extensions.push_back(".TIF");
            image_extensions.push_back(".ppm");
            image_extensions.push_back(".PPM");
            image_extensions.push_back(".bmp");
            image_extensions.push_back(".BMP");

            for (int i = 0; i < image_extensions.size(); i++)
            {
                if (filename.find(image_extensions.at(i)) != std::string::npos)
                {
                    return true;
                }
            }
            return false;
        }

        inline int getImageList(std::string dir, std::vector<std::string> &files)
        {
            DIR *dp;
            struct dirent *dirp;
            if ((dp = opendir(dir.c_str())) == NULL)
            {
                return -1;
            }

            while ((dirp = readdir(dp)) != NULL)
            {
                std::string name = std::string(dirp->d_name);
                if (name != "." && name != ".." && isimage(name))
                {
                    files.push_back(name);
                }
            }
            closedir(dp);

            std::sort(files.begin(), files.end());
            return files.size();
        }
    } // namespace Utils
} // namespace SLAM

#endif