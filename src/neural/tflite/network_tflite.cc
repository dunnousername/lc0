/*
 This file is part of Leela Chess Zero.
 Copyright (C) 2018-2019 The LCZero Authors
 Leela Chess is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 Leela Chess is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 You should have received a copy of the GNU General Public License
 along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "neural/factory.h"
#include "neural/loader.h"
#include "neural/network.h"
#include "neural/network_legacy.h"

namespace lczero {
  using namespace tflite_backend;
  class TFLiteNetwork : public Network {
    public: 
     TFLiteNetwork(const WeightsFile& file) {
      this->file = file;
      this->model = tflite::FlatBufferModel::BuildFromFile(filename?);
      tflite::ops::builtin::BuiltinOpResolver resolver;
      //std::unique_ptr<tflite::Interpreter> interpreter;
      tflite::InterpreterBuilder(*model, resolver)(&this->interpreter);
     }
     WeightsFile& file;
     tflite::FlatBufferModel& model;
     std::unique_ptr<tflite::Interpreter> interpreter;
    private:
     
  };
  class TFLiteComputation : public NetworkComputation {
    public:
     TFLiteComputation(const TFLiteNetwork *network) {
      this->network = network;
     }
     virtual ~TFLiteComputation() {}
     void AddInput(InputPlanes&& input) override { planes_.emplace_back(input); }
     void ComputeBlocking() override {
      
     }
     int GetBatchSize() const override { return static_cast<int>(planes_.size()); }
     
    private:
     std::vector<InputPlanes> planes_;
     TFLiteNetwork *network;
  };
}
