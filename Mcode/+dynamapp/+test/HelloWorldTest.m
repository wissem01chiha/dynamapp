classdef HelloWorldTest < matlab.unittest.TestCase

  methods (Test)

    function testHelloWorld(t)
      fprintf('Hello, World!')
      t.verifyNotEmpty('Hello, World!')
    end

  end

end
